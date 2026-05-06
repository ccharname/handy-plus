// MLXAudioBridge — C-callable FFI bridge for mlx-audio-swift STT
//
// Mirrors the pattern in handy-plus apple_speech.swift / apple_intelligence.swift:
//   - @_cdecl functions are visible to Rust as extern "C"
//   - All returned heap strings are strdup'd; caller frees via mlx_audio_bridge_free_string
//
// P1 metallib verification (2026-05-04):
//   The bare CLI probe at /tmp/handy-mlx-swift-poc/ showed:
//     [mlx-bridge] Error: MLX error: Failed to load the default metallib.
//   This is expected for a CLI binary with no bundle. The fix is the Tauri .app bundle:
//   `bun run tauri build` uses Xcode's SPM integration which compiles .metal shaders and
//   bundles default.metallib inside the .app automatically (Resources/default.metallib).
//   Verification: run the app, trigger mlx_audio_bridge_version() call on startup —
//   any metallib crash would manifest as SIGABRT. If that call succeeds, P1 is confirmed.
//
// Swift 6 fixes applied vs PoC:
//   1. SyncBox<T> defined at module scope (not inside generic function) — Swift 6
//      disallows generic class nested in generic function.
//   2. Closure passed to Task {} marked @Sendable to satisfy strict concurrency.

import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT

// MARK: - Swift 6 async-to-sync helper
//
// Defined at module scope (not inside a generic function) to avoid:
//   "type cannot be nested in generic function" — Swift 6 regression vs Swift 5.
private final class SyncBox<T>: @unchecked Sendable {
    var value: T?
    var error: Error?
}

/// Lightweight @unchecked Sendable value box for crossing Task concurrency boundaries.
///
/// Used in place of the identically-named type defined inside MLXAudioSTT (which is
/// not exported from the mlx-audio-swift module into this bridge target).
private struct BridgeSendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// Runs an async closure synchronously by blocking a DispatchSemaphore.
///
/// The async-to-sync dance is non-trivial when this bridge is loaded into a
/// non-Swift host process (e.g. a Rust binary via static link): Swift's
/// cooperative thread pool does NOT auto-initialise without a Swift `main` /
/// SwiftUI App entry, so `Task { ... }` and `Task.detached { ... }` both sit
/// on an empty queue and never run — the calling thread blocks on the
/// semaphore forever.
///
/// Workaround: spin up a dedicated pthread with its own RunLoop, then submit
/// the async work to a serial DispatchQueue inside that thread. The RunLoop
/// gives Swift Concurrency the dispatch context it needs to actually schedule
/// the Task. The outer semaphore bridges the result back to the caller.
private func runSync<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) throws -> T {
    let box = SyncBox<T>()
    let outerSem = DispatchSemaphore(value: 0)

    let workerThread = Thread {
        autoreleasepool {
            let innerSem = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) { @Sendable in
                do { box.value = try await body() }
                catch { box.error = error }
                innerSem.signal()
            }
            // Drive the RunLoop in 50 ms slices until the Task completes.
            // RunLoop activity gives Swift Concurrency a host context so the
            // Task is actually scheduled.
            //
            // 30-minute hard ceiling: covers the worst-case "first use"
            // path where mlx-audio-swift downloads ~3.5 GB of Voxtral
            // weights to its cache subdir before transcribing. After
            // weights are cached, real-world inference is sub-30 s on
            // M-series.
            let deadline = Date(timeIntervalSinceNow: 1800)
            while innerSem.wait(timeout: .now() + .milliseconds(10)) == .timedOut {
                RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
                if Date() > deadline {
                    box.error = BridgeError.timeout
                    break
                }
            }
            outerSem.signal()
        }
    }
    workerThread.qualityOfService = .userInitiated
    workerThread.name = "mlx-audio-bridge-runner"
    workerThread.start()

    outerSem.wait()
    if let error = box.error { throw error }
    return box.value!
}

// MARK: - Version probe

/// Returns the bridge + mlx-audio-swift version string as a strdup'd C string.
/// Callable from Rust with no model loaded — used as a smoke test that the library
/// linked correctly and (in Tauri .app context) that Metal initialises without error.
///
/// Caller must free with mlx_audio_bridge_free_string().
@_cdecl("mlx_audio_bridge_version")
public func mlxAudioBridgeVersion() -> UnsafeMutablePointer<CChar> {
    let version = "handy-mlx-bridge/0.1.0 mlx-audio-swift/0.1.2"
    return strdup(version)!
}

// MARK: - String memory management

/// Free a C string returned by any mlx_audio_bridge_* function.
/// Must be called for every non-nil pointer returned by this bridge.
@_cdecl("mlx_audio_bridge_free_string")
public func mlxAudioBridgeFreeString(_ ptr: UnsafeMutablePointer<CChar>?) {
    guard let ptr else { return }
    free(ptr)
}

// MARK: - One-shot file transcription

/// Transcribe a WAV file using an mlx-community HuggingFace model.
///
/// Parameters:
///   - wavPath: UTF-8 NUL-terminated path to a 16 kHz mono WAV file.
///   - modelId: Logical model identifier string. Supported values:
///       "voxtral-mini-4b-4bit"  → mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
///       "qwen3-asr-06b-8bit"    → mlx-community/Qwen3-ASR-0.6B-8bit   (reserved)
///   - outText: On success, set to a strdup'd transcription string. Caller frees.
///   - outError: On failure, set to a strdup'd error message. Caller frees. nil on success.
///
/// Returns: 0 on success, non-zero on error.
///
/// Thread safety: This function blocks the calling thread (via DispatchSemaphore) until
/// transcription completes. Do NOT call from the main thread in a UI application.
/// Call from a dedicated Rust worker thread (which is the default Tauri handler context).
@_cdecl("mlx_audio_transcribe_file")
public func mlxAudioTranscribeFile(
    _ wavPath: UnsafePointer<CChar>,
    _ modelId: UnsafePointer<CChar>,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    outText.pointee = nil
    outError.pointee = nil

    let wavPathStr = String(cString: wavPath)
    let modelIdStr = String(cString: modelId)

    // Map logical model id to HuggingFace repo path.
    let repoId: String
    switch modelIdStr {
    case "voxtral-mini-4b-4bit":
        repoId = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    case "qwen3-asr-06b-8bit":
        repoId = "mlx-community/Qwen3-ASR-0.6B-8bit"
    default:
        // Pass through unknown ids directly (allows testing with explicit repo paths).
        repoId = modelIdStr
    }

    do {
        let text = try runSync { () async throws -> String in
            let audioURL = URL(fileURLWithPath: wavPathStr)
            guard FileManager.default.fileExists(atPath: wavPathStr) else {
                throw BridgeError.fileNotFound(wavPathStr)
            }

            // Load audio file and ensure 16 kHz mono — mlx-audio-swift requirement.
            let (inputSampleRate, inputAudio) = try loadAudioArray(from: audioURL)
            let audio: MLXArray
            if inputSampleRate != 16000 {
                audio = try resampleAudio(inputAudio, from: inputSampleRate, to: 16000)
            } else {
                // Force mono if multi-channel by averaging across the last axis.
                audio = inputAudio.ndim > 1 ? inputAudio.mean(axis: -1) : inputAudio
            }

            // Load model from HuggingFace cache (or download on first use).
            let model = try await loadSTTModel(repo: repoId)

            // Generation parameters — use greedy decode (temperature=0) for determinism.
            // maxTokens=4096 covers ~3000 Chinese characters (30 min of speech).
            // FD-003 M3.5 #5: language="Chinese" skips Qwen3-ASR auto-detection,
            // saving ~20-50ms first_token_ms. "Chinese" maps via support_languages
            // config (case-insensitive). Single-language (zh) users: no precision loss.
            let params = STTGenerateParameters(
                maxTokens: 4096,
                temperature: 0.0,
                verbose: false,
                language: "Chinese"
            )

            // Non-streaming inference: collect the final Result event.
            var finalText = ""
            for try await event in model.generateStream(audio: audio, generationParameters: params) {
                switch event {
                case .result(let output):
                    finalText = output.text
                default:
                    break
                }
            }
            return finalText
        }

        outText.pointee = strdup(text)
        return 0
    } catch {
        let msg = "\(error)"
        outError.pointee = strdup(msg)
        return 1
    }
}

// MARK: - Sendable wrapper for C pointer pair (Swift 6 strict concurrency)
//
// `UnsafeMutableRawPointer?` and `@convention(c)` function pointers are not
// `Sendable` in Swift 6.  We need to ferry them across Task boundaries inside
// `runSync`.  Marking the box `@unchecked Sendable` is safe here because:
//   1. The pointers originate from the Rust caller and are guaranteed valid for
//      the entire duration of the `runSync` call (which blocks the Rust thread
//      via DispatchSemaphore until the async work finishes).
//   2. We never store or alias the pointers beyond the scope of the Task.
private struct StreamingCallbackContext: @unchecked Sendable {
    let tokenCb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void
    let ctx: UnsafeMutableRawPointer?
}

// MARK: - Streaming file transcription

/// Transcribe a WAV file using an mlx-community model, emitting cumulative partial
/// text via a C callback on every decoded token.
///
/// Parameters:
///   - wavPath: UTF-8 NUL-terminated path to a 16 kHz mono WAV file.
///   - modelId: Logical model identifier (same as mlxAudioTranscribeFile).
///   - tokenCb: Called on every token with (cumulative_text_cstring, ctx).
///              The cstring is valid only for the duration of the callback.
///              On the final `.result` event, tokenCb is also called with the
///              full authoritative text (may differ from last cumulative partial
///              due to whitespace trimming).
///   - ctx: Opaque context pointer forwarded to every tokenCb call.
///   - errorOut: On failure, set to a strdup'd error message. Caller frees via
///              mlx_audio_bridge_free_string. nil on success.
///
/// Returns: 0 on success, non-zero on error.
///
/// Thread safety: blocks the calling thread (via DispatchSemaphore).
/// Call from a dedicated Rust worker thread only.
@_cdecl("mlx_audio_transcribe_streaming")
public func mlxAudioTranscribeStreaming(
    _ wavPath: UnsafePointer<CChar>,
    _ modelId: UnsafePointer<CChar>,
    _ tokenCb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void,
    _ ctx: UnsafeMutableRawPointer?,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    errorOut?.pointee = nil

    let wavPathStr = String(cString: wavPath)
    let modelIdStr = String(cString: modelId)

    // Map logical model id to HuggingFace repo path (same mapping as batch path).
    let repoId: String
    switch modelIdStr {
    case "voxtral-mini-4b-4bit":
        repoId = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    case "qwen3-asr-06b-8bit":
        repoId = "mlx-community/Qwen3-ASR-0.6B-8bit"
    default:
        repoId = modelIdStr
    }

    // Wrap the C pointers in @unchecked Sendable so they can cross the Task boundary.
    let cbCtx = StreamingCallbackContext(tokenCb: tokenCb, ctx: ctx)

    do {
        try runSync { () async throws -> Void in
            let audioURL = URL(fileURLWithPath: wavPathStr)
            guard FileManager.default.fileExists(atPath: wavPathStr) else {
                throw BridgeError.fileNotFound(wavPathStr)
            }

            // Load and normalise audio to 16 kHz mono.
            let (inputSampleRate, inputAudio) = try loadAudioArray(from: audioURL)
            let audio: MLXArray
            if inputSampleRate != 16000 {
                audio = try resampleAudio(inputAudio, from: inputSampleRate, to: 16000)
            } else {
                audio = inputAudio.ndim > 1 ? inputAudio.mean(axis: -1) : inputAudio
            }

            // Load model (from HF cache).
            let model = try await loadSTTModel(repo: repoId)

            // FD-003 M3.5 #5: language hint — same as batch path above.
            let params = STTGenerateParameters(
                maxTokens: 4096,
                temperature: 0.0,
                verbose: false,
                language: "Chinese"
            )

            // Accumulate tokens into cumulative partial; emit callback on every token.
            var accumulated = ""
            for try await event in model.generateStream(audio: audio, generationParameters: params) {
                switch event {
                case .token(let tokenText):
                    accumulated += tokenText
                    // Emit cumulative partial — cstring is valid inside withCString block.
                    accumulated.withCString { cstr in
                        cbCtx.tokenCb(cstr, cbCtx.ctx)
                    }
                case .result(let output):
                    // Emit final authoritative text (trimmed, post-processed).
                    // This may differ from last accumulated partial by whitespace.
                    let finalText = output.text
                    finalText.withCString { cstr in
                        cbCtx.tokenCb(cstr, cbCtx.ctx)
                    }
                default:
                    break
                }
            }
        }
        return 0
    } catch {
        let msg = "\(error)"
        errorOut?.pointee = strdup(msg)
        return 1
    }
}

// MARK: - Qwen3-ASR lexical biasing helpers (FD-009 M3)

/// Build a Qwen3-ASR prompt with an optional lexical biasing context injected into
/// the system message.
///
/// The Qwen3-ASR paper (arxiv 2601.21337 §3.1) demonstrates 30-60% relative entity-WER
/// improvement by providing entity lists in the system turn.  This function produces the
/// same token sequence as `Qwen3ASRModel.buildPrompt` when `context` is nil/empty, and
/// inserts the context string as system-message content otherwise.
///
/// Format when context is non-empty:
///   <|im_start|>system
///   <context>
///   <|im_end|>
///
/// - Parameters:
///   - model: A loaded `Qwen3ASRModel` with tokenizer set.
///   - numAudioTokens: Number of `<|audio_pad|>` placeholder tokens.
///   - language: Human-readable language name (e.g. "Chinese").
///   - context: Optional system-message context for biasing.  When nil or empty,
///              behaviour is identical to `Qwen3ASRModel.buildPrompt`.
/// - Returns: 2-D input-ids tensor `[1, seqLen]`.
private func buildQwen3PromptWithContext(
    model: Qwen3ASRModel,
    numAudioTokens: Int,
    language: String,
    context: String?
) -> MLXArray {
    guard let tokenizer = model.tokenizer else {
        fatalError("Qwen3ASRModel tokenizer not loaded")
    }

    // Resolve canonical language name (mirrors Qwen3ASRModel.buildPrompt).
    let supported = model.config.supportLanguages
    let supportedLower = Dictionary(uniqueKeysWithValues: supported.map { ($0.lowercased(), $0) })
    let langName = supportedLower[language.lowercased()] ?? language

    // Build system turn: empty when no context, otherwise inject context text.
    let systemContent: String
    if let ctx = context, !ctx.isEmpty {
        systemContent = ctx + "\n"
    } else {
        systemContent = ""
    }

    let prompt = "<|im_start|>system\n\(systemContent)<|im_end|>\n"
        + "<|im_start|>user\n<|audio_start|>"
        + String(repeating: "<|audio_pad|>", count: numAudioTokens)
        + "<|audio_end|><|im_end|>\n"
        + "<|im_start|>assistant\nlanguage \(langName)<asr_text>"

    let tokenIds = tokenizer.encode(text: prompt)
    return MLXArray(tokenIds.map { Int32($0) }).expandedDimensions(axis: 0)
}

/// Run streaming generation on a `Qwen3ASRModel` with optional lexical biasing context.
///
/// Mirrors `Qwen3ASRModel.generateStream` but substitutes the `buildPrompt` call with
/// `buildQwen3PromptWithContext`, allowing system-message injection without forking
/// mlx-audio-swift.  When `context` is nil the output is identical to calling
/// `model.generateStream(audio:generationParameters:)` directly.
private func qwen3GenerateStreamWithContext(
    model: Qwen3ASRModel,
    audio: MLXArray,
    generationParameters: STTGenerateParameters,
    context: String?
) -> AsyncThrowingStream<STTGeneration, Error> {
    // Wrap all non-Sendable values into @unchecked Sendable boxes.
    let sendableModel = BridgeSendableBox(model)
    let sendableAudio = BridgeSendableBox(audio)
    let sendableContext = BridgeSendableBox(context)
    let sendableParams = BridgeSendableBox(generationParameters)

    return AsyncThrowingStream { continuation in
        Task.detached {
            let m = sendableModel.value
            let aud = sendableAudio.value
            let ctx = sendableContext.value
            let params = sendableParams.value
            do {
                guard let tokenizer = m.tokenizer else {
                    throw STTError.modelNotInitialized("Tokenizer not loaded")
                }

                let startTime = Date()
                let eosTokenIds = [151645, 151643]

                let chunks = splitAudioIntoChunks(
                    aud,
                    sampleRate: m.sampleRate,
                    chunkDuration: params.chunkDuration,
                    minChunkDuration: params.minChunkDuration
                )

                var totalPromptTokens = 0
                var totalGenerationTokens = 0
                var remainingTokens = params.maxTokens
                var allGeneratedTokens: [Int] = []

                for (chunkAudio, _) in chunks {
                    if remainingTokens <= 0 { break }
                    try Task.checkCancellation()

                    let (inputFeatures, featureAttentionMask, numAudioTokens) = m.preprocessAudio(chunkAudio)
                    // Context-aware prompt: replaces m.buildPrompt(...).
                    let inputIds = buildQwen3PromptWithContext(
                        model: m,
                        numAudioTokens: numAudioTokens,
                        language: params.language,
                        context: ctx
                    )
                    let promptTokenCount = inputIds.dim(1)
                    totalPromptTokens += promptTokenCount

                    // Use callAsFunction with inputFeatures on the first (prefill) pass.
                    // Qwen3ASRModel.callAsFunction handles embedTokens + mergeAudioFeatures
                    // internally when inputEmbeddings is nil and inputFeatures is non-nil,
                    // avoiding the need to access the internal `model.embedTokens` property.
                    let cache = m.makeCache()
                    var logits = m.callAsFunction(
                        inputIds: inputIds,
                        inputFeatures: inputFeatures,
                        featureAttentionMask: featureAttentionMask,
                        cache: cache
                    )
                    MLX.eval(logits)

                    var chunkTokens: [Int] = []

                    for _ in 0..<remainingTokens {
                        try Task.checkCancellation()

                        var lastLogits = logits[0..., -1, 0...]
                        if params.temperature > 0 {
                            lastLogits = lastLogits / params.temperature
                        }
                        let nextToken = lastLogits.argMax(axis: -1).item(Int.self)

                        if eosTokenIds.contains(nextToken) {
                            break
                        }

                        chunkTokens.append(nextToken)
                        allGeneratedTokens.append(nextToken)

                        let tokenText = tokenizer.decode(tokens: [nextToken])
                        continuation.yield(.token(tokenText))

                        let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
                        logits = m.callAsFunction(inputIds: nextTokenArray, cache: cache)
                        MLX.eval(logits)
                    }

                    totalGenerationTokens += chunkTokens.count
                    remainingTokens -= chunkTokens.count

                    Memory.clearCache()
                }

                let endTime = Date()
                let totalTime = endTime.timeIntervalSince(startTime)
                let tokensPerSecond = totalTime > 0 ? Double(totalGenerationTokens) / totalTime : 0
                let peakMemory = Double(Memory.peakMemory) / 1e9

                let info = STTGenerationInfo(
                    promptTokenCount: totalPromptTokens,
                    generationTokenCount: totalGenerationTokens,
                    prefillTime: 0,
                    generateTime: totalTime,
                    tokensPerSecond: tokensPerSecond,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.info(info))

                let text = tokenizer.decode(tokens: allGeneratedTokens)
                let output = STTOutput(
                    text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                    promptTokens: totalPromptTokens,
                    generationTokens: totalGenerationTokens,
                    totalTokens: totalPromptTokens + totalGenerationTokens,
                    promptTps: totalTime > 0 ? Double(totalPromptTokens) / totalTime : 0,
                    generationTps: tokensPerSecond,
                    totalTime: totalTime,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.result(output))
                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}

// MARK: - Streaming file transcription with lexical biasing context (FD-009 M3)

/// Transcribe a WAV file using an mlx-community model with optional lexical biasing.
///
/// Identical to `mlx_audio_transcribe_streaming` except for the additional `context`
/// parameter.  When `context` is non-NULL and non-empty, it is injected into the
/// Qwen3-ASR system message to bias the decoder towards domain-specific vocabulary.
/// For non-Qwen3-ASR models the context is silently ignored (falls back to standard
/// `generateStream`).
///
/// Parameters:
///   - wavPath:    UTF-8 NUL-terminated path to a 16 kHz mono WAV file.
///   - modelId:    Logical model identifier (same as other streaming functions).
///   - context:    NUL-terminated UTF-8 context string (may be NULL or empty string
///                 to disable biasing).  Typically ~60-1500 chars of comma-separated
///                 entity / word list.
///   - tokenCb:    Callback called on every decoded token with cumulative text.
///   - ctx:        Opaque context pointer forwarded to tokenCb.
///   - errorOut:   On failure, set to a strdup'd error. Caller frees via
///                 mlx_audio_bridge_free_string. nil on success.
///
/// Returns: 0 on success, non-zero on error.
///
/// Thread safety: blocks the calling thread (via DispatchSemaphore).
@_cdecl("mlx_audio_transcribe_streaming_with_context")
public func mlxAudioTranscribeStreamingWithContext(
    _ wavPath: UnsafePointer<CChar>,
    _ modelId: UnsafePointer<CChar>,
    _ context: UnsafePointer<CChar>?,
    _ tokenCb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void,
    _ ctx: UnsafeMutableRawPointer?,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    errorOut?.pointee = nil

    let wavPathStr = String(cString: wavPath)
    let modelIdStr = String(cString: modelId)
    let contextStr: String? = context.map { String(cString: $0) }
        .flatMap { $0.isEmpty ? nil : $0 }

    let repoId: String
    switch modelIdStr {
    case "voxtral-mini-4b-4bit":
        repoId = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    case "qwen3-asr-06b-8bit":
        repoId = "mlx-community/Qwen3-ASR-0.6B-8bit"
    default:
        repoId = modelIdStr
    }

    let cbCtx = StreamingCallbackContext(tokenCb: tokenCb, ctx: ctx)
    let sendableContext = BridgeSendableBox(contextStr)

    do {
        try runSync { () async throws -> Void in
            let audioURL = URL(fileURLWithPath: wavPathStr)
            guard FileManager.default.fileExists(atPath: wavPathStr) else {
                throw BridgeError.fileNotFound(wavPathStr)
            }

            let (inputSampleRate, inputAudio) = try loadAudioArray(from: audioURL)
            let audio: MLXArray
            if inputSampleRate != 16000 {
                audio = try resampleAudio(inputAudio, from: inputSampleRate, to: 16000)
            } else {
                audio = inputAudio.ndim > 1 ? inputAudio.mean(axis: -1) : inputAudio
            }

            let params = STTGenerateParameters(
                maxTokens: 4096,
                temperature: 0.0,
                verbose: false,
                language: "Chinese"
            )

            // Load the model; use context-aware path for Qwen3-ASR, fall through
            // to standard generateStream for all other model types.
            let resolvedContext = sendableContext.value

            // Try Qwen3-ASR context path first (only model that supports biasing).
            if repoId.lowercased().contains("qwen3-asr") || repoId.lowercased().contains("qwen3_asr") {
                let qwen3Model = try await Qwen3ASRModel.fromPretrained(repoId)
                var accumulated = ""
                for try await event in qwen3GenerateStreamWithContext(
                    model: qwen3Model,
                    audio: audio,
                    generationParameters: params,
                    context: resolvedContext
                ) {
                    switch event {
                    case .token(let tokenText):
                        accumulated += tokenText
                        accumulated.withCString { cstr in
                            cbCtx.tokenCb(cstr, cbCtx.ctx)
                        }
                    case .result(let output):
                        let finalText = output.text
                        finalText.withCString { cstr in
                            cbCtx.tokenCb(cstr, cbCtx.ctx)
                        }
                    default:
                        break
                    }
                }
            } else {
                // Non-Qwen3-ASR models: ignore context, use standard path.
                let model = try await loadSTTModel(repo: repoId)
                var accumulated = ""
                for try await event in model.generateStream(audio: audio, generationParameters: params) {
                    switch event {
                    case .token(let tokenText):
                        accumulated += tokenText
                        accumulated.withCString { cstr in
                            cbCtx.tokenCb(cstr, cbCtx.ctx)
                        }
                    case .result(let output):
                        let finalText = output.text
                        finalText.withCString { cstr in
                            cbCtx.tokenCb(cstr, cbCtx.ctx)
                        }
                    default:
                        break
                    }
                }
            }
        }
        return 0
    } catch {
        let msg = "\(error)"
        errorOut?.pointee = strdup(msg)
        return 1
    }
}

// MARK: - Model loader

/// Route a HuggingFace repo id to the appropriate mlx-audio-swift STTGenerationModel.
/// Uses heuristic name matching — same approach as the official mlx-audio-swift CLI.
private func loadSTTModel(repo: String) async throws -> any STTGenerationModel {
    let lower = repo.lowercased()
    if lower.contains("voxtral") {
        return try await VoxtralRealtimeModel.fromPretrained(repo)
    }
    if lower.contains("qwen3-asr") || lower.contains("qwen3_asr") {
        return try await Qwen3ASRModel.fromPretrained(repo)
    }
    if lower.contains("glmasr") || lower.contains("glm-asr") {
        return try await GLMASRModel.fromPretrained(repo)
    }
    if lower.contains("parakeet") {
        return try await ParakeetModel.fromPretrained(repo)
    }
    // Note: SenseVoiceModel is not included in mlx-audio-swift v0.1.2.
    // If the repo name contains "sensevoice", fall through to Qwen3-ASR as default.
    // Default: treat as Qwen3-ASR (broad multilingual fallback).
    return try await Qwen3ASRModel.fromPretrained(repo)
}

// MARK: - Bridge errors

private enum BridgeError: Error {
    case fileNotFound(String)
    case unsupportedModel(String)
    case timeout
}
