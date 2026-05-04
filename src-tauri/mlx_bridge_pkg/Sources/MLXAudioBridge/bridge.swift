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

/// Runs an async closure synchronously by blocking a DispatchSemaphore.
/// Required because @_cdecl cannot be async, but mlx-audio-swift uses async/await.
/// The @Sendable annotation satisfies Swift 6 strict concurrency checks.
private func runSync<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) throws -> T {
    let box = SyncBox<T>()
    let sem = DispatchSemaphore(value: 0)
    Task { @Sendable in
        do { box.value = try await body() }
        catch { box.error = error }
        sem.signal()
    }
    sem.wait()
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
            // language="" passes empty string which both Voxtral and Qwen3-ASR treat as
            // auto-detect (the models inspect the audio spectrogram for language cues).
            let params = STTGenerateParameters(
                maxTokens: 4096,
                temperature: 0.0,
                verbose: false,
                language: ""    // auto-detect; empty string → model-level language inference
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
}
