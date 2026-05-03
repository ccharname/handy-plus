import AVFoundation
import Dispatch
import Foundation
import Speech

// MARK: - Swift implementation for Apple Speech integration
// This file is compiled via Cargo build script for macOS targets (aarch64 + x86_64).
//
// DONE(macOS-26): SFSpeechRecognizer is soft-deprecated on macOS 26 (Tahoe) in favour
// of the new SpeechAnalyzer + SpeechTranscriber APIs introduced at WWDC 2025.
// This file implements BOTH paths with an #available(macOS 26.0, *) gate:
//   • macOS 26+ → transcribeImplSpeechAnalyzer (SpeechAnalyzer + SpeechTranscriber)
//   • macOS 10.15–25 → transcribeImplLegacy (SFSpeechRecognizer, unchanged)
//
// The @_cdecl FFI surface is IDENTICAL between the two paths — Rust callers are
// oblivious to which engine ran.
// API reference: https://developer.apple.com/documentation/speech/speechanalyzer

private typealias ResponsePointer = UnsafeMutablePointer<AppleSpeechResponse>

private func duplicateCString(_ text: String) -> UnsafeMutablePointer<CChar>? {
    return text.withCString { basePointer in
        guard let duplicated = strdup(basePointer) else {
            return nil
        }
        return duplicated
    }
}

// MARK: - Availability check

@_cdecl("is_apple_speech_available")
public func isAppleSpeechAvailable() -> Int32 {
    guard #available(macOS 10.15, *) else {
        return 0
    }
    // Check that at least one on-device recognizer is available by querying
    // the supported locales. We specifically verify on-device availability
    // using the default locale as a probe.
    let supported = SFSpeechRecognizer.supportedLocales()
    if supported.isEmpty {
        return 0
    }
    // Verify at least one recognizer can be instantiated
    let probe = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    guard probe != nil else {
        return 0
    }
    return 1
}

// MARK: - Transcription (internal implementation — dispatches to new or legacy path)

/// Shared implementation for both the plain and with-partials variants.
/// On macOS 26+, dispatches to `transcribeImplSpeechAnalyzer`.
/// On macOS 10.15-25, falls back to `transcribeImplLegacy` (SFSpeechRecognizer).
/// `onPartial` is called for each non-final result; pass nil to disable.
@available(macOS 10.15, *)
private func transcribeImpl(
    samples: UnsafePointer<Float>,
    sampleCount: Int,
    sampleRate: Double,
    localeBcp47: UnsafePointer<CChar>,
    contextualStrings: UnsafePointer<UnsafePointer<CChar>?>?,
    contextualCount: Int,
    requireOnDevice: Int32,
    timeoutMs: Int32,
    onPartial: ((String) -> Void)?
) -> UnsafeMutablePointer<AppleSpeechResponse> {
    // TEMPORARILY DISABLED: SpeechAnalyzer path crashes with SIGTRAP at
    // SpeechRecognizerWorker.preRunRecognition() — root cause is the per-locale
    // model has not been pre-downloaded via SpeechTranscriber.downloadModel(for:),
    // and Apple's worker fatal-errors when it tries to run an uninstalled model.
    //
    // Crash: handy-2026-05-03-231748.ips, EXC_BREAKPOINT in cooperative queue.
    //
    // To re-enable, the SpeechAnalyzer path must first await
    // `SpeechTranscriber.supportedLocales(for: .transcription)` then
    // `downloadModel(for: locale)` if not already installed. Until that flow
    // is wired, force-fall through to the legacy SFSpeechRecognizer path.
    //
    // if #available(macOS 26.0, *) {
    //     return transcribeImplSpeechAnalyzer(...)
    // }
    return transcribeImplLegacy(
        samples: samples,
        sampleCount: sampleCount,
        sampleRate: sampleRate,
        localeBcp47: localeBcp47,
        contextualStrings: contextualStrings,
        contextualCount: contextualCount,
        requireOnDevice: requireOnDevice,
        timeoutMs: timeoutMs,
        onPartial: onPartial
    )
}

// MARK: - New path: SpeechAnalyzer + SpeechTranscriber (macOS 26+)

/// Implements transcription using the new SpeechAnalyzer / SpeechTranscriber API
/// (macOS 26+, WWDC 2025).
///
/// Design notes:
///   • The async-sequence-based API is bridged back to the synchronous @_cdecl ABI
///     via a DispatchSemaphore + GCD timeout guard, matching the legacy path pattern.
///   • `SpeechTranscriber.Preset.progressiveTranscription` enables volatile (partial)
///     results in addition to final results. For the no-partial variant we use the
///     simpler `.transcription` preset.
///   • `result.isFinal` (from SpeechModuleResult) distinguishes volatile vs final.
///   • Contextual strings are passed via `AnalysisContext.contextualStrings[.general]`.
///   • `requireOnDevice` has no direct equivalent — SpeechAnalyzer on macOS 26 is
///     always on-device; the flag is accepted but silently ignored.
///   • `taskHint` / `addsPunctuation` have no direct equivalents in the new API;
///     punctuation and capitalization are on by default in the new engine.
@available(macOS 26.0, *)
private func transcribeImplSpeechAnalyzer(
    samples: UnsafePointer<Float>,
    sampleCount: Int,
    sampleRate: Double,
    localeBcp47: UnsafePointer<CChar>,
    contextualStrings: UnsafePointer<UnsafePointer<CChar>?>?,
    contextualCount: Int,
    requireOnDevice: Int32,  // NOTE: ignored — SpeechAnalyzer is always on-device
    timeoutMs: Int32,
    onPartial: ((String) -> Void)?
) -> UnsafeMutablePointer<AppleSpeechResponse> {
    print("[apple_speech] Engine: SpeechAnalyzer (macOS 26+)")

    let responsePtr = ResponsePointer.allocate(capacity: 1)
    responsePtr.initialize(to: AppleSpeechResponse(text: nil, success: 0, error_message: nil))

    let localeStr = String(cString: localeBcp47)
    let locale = Locale(identifier: localeStr)

    // Build contextual strings array
    var contextualArray: [String] = []
    if let ptr = contextualStrings, contextualCount > 0 {
        for i in 0..<contextualCount {
            if let cstr = ptr[i] {
                contextualArray.append(String(cString: cstr))
            }
        }
    }

    // Thread-safe container to pass results from async Task back to calling thread
    final class ResultBox: @unchecked Sendable {
        var text: String?
        var error: String?
    }
    let box = ResultBox()
    let semaphore = DispatchSemaphore(value: 0)

    // Build AVAudioFormat for the input PCM buffer
    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: sampleRate,
        channels: 1,
        interleaved: false
    ) else {
        responsePtr.pointee.error_message = duplicateCString(
            "Failed to create AVAudioFormat for sample rate \(sampleRate)."
        )
        return responsePtr
    }

    guard let pcmBuffer = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(sampleCount)
    ) else {
        responsePtr.pointee.error_message = duplicateCString(
            "Failed to allocate AVAudioPCMBuffer."
        )
        return responsePtr
    }

    pcmBuffer.frameLength = AVAudioFrameCount(sampleCount)
    if let channelData = pcmBuffer.floatChannelData {
        channelData[0].update(from: samples, count: sampleCount)
    }

    // Choose preset: progressiveTranscription enables volatile (partial) results;
    // plain transcription only emits final results.
    let preset: SpeechTranscriber.Preset = onPartial != nil
        ? .progressiveTranscription
        : .transcription

    let transcriber = SpeechTranscriber(locale: locale, preset: preset)

    // Wire contextual strings via AnalysisContext
    let analysisContext = AnalysisContext()
    if !contextualArray.isEmpty {
        analysisContext.contextualStrings[.general] = contextualArray
    }

    // Create the analyzer (modules-only init; we feed via start(inputSequence:))
    let analyzer = SpeechAnalyzer(
        modules: [transcriber],
        options: nil
    )

    // Launch the async task that drives the analyzer and collects results.
    // The Task bridges the async API back to the semaphore-based synchronous ABI.
    //
    // Pattern: two concurrent child tasks via withTaskGroup —
    //   1. "feeder" task: feeds the single PCM buffer and calls analyzeSequence,
    //      which internally drives the analysis pipeline to completion.
    //   2. "collector" task: iterates transcriber.results, delivering partials and
    //      capturing the last final result.
    // The group awaits both before returning; if either throws, the group cancels.
    let task = Task {
        do {
            // Set context before starting so contextual strings are applied.
            try await analyzer.setContext(analysisContext)

            // Build an AsyncStream that yields our single PCM buffer and then ends.
            let inputStream = AsyncStream<AnalyzerInput> { continuation in
                continuation.yield(AnalyzerInput(buffer: pcmBuffer))
                continuation.finish()
            }

            // Run feeding and result collection concurrently.
            var lastFinalText: String? = nil
            try await withThrowingTaskGroup(of: String?.self) { group in
                // Child 1: feed audio and drive analysis
                group.addTask {
                    // analyzeSequence feeds input AND waits for all analysis to complete.
                    // It signals the transcriber.results stream to finish when done.
                    _ = try await analyzer.analyzeSequence(inputStream)
                    return nil
                }

                // Child 2: collect transcription results
                group.addTask { [onPartial] in
                    var lastText: String? = nil
                    for try await result in transcriber.results {
                        // Extract plain text from AttributedString
                        let plainText = String(result.text.characters)
                        if result.isFinal {
                            lastText = plainText
                        } else {
                            // Volatile (partial) result — deliver to callback if provided
                            onPartial?(plainText)
                        }
                    }
                    return lastText
                }

                // Collect results from both tasks
                for try await result in group {
                    if let text = result {
                        lastFinalText = text
                    }
                }
            }

            if let text = lastFinalText {
                box.text = text
            } else {
                // No final result emitted — treat as empty transcription (silence)
                box.text = ""
            }
        } catch {
            box.error = "ENGINE: \(error.localizedDescription)"
        }
        semaphore.signal()
    }

    // GCD timer guard: independently cancels the Task and signals the semaphore
    // after timeoutMs, mirroring the legacy path's safety net.
    var timerWorkItem: DispatchWorkItem?
    if timeoutMs > 0 {
        let workItem = DispatchWorkItem {
            task.cancel()
            if box.text == nil && box.error == nil {
                box.error = "TIMEOUT: Apple Speech (SpeechAnalyzer) timed out after \(timeoutMs)ms. The recognizer may be unavailable or waiting for first-use initialization."
            }
            semaphore.signal()
        }
        timerWorkItem = workItem
        DispatchQueue.global(qos: .userInitiated).asyncAfter(
            deadline: .now() + .milliseconds(Int(timeoutMs)),
            execute: workItem
        )
    }

    // Wait — either the Task or the GCD timer will signal us.
    semaphore.wait()

    // Cancel the timer if the Task finished first (cancelling an already-executed
    // DispatchWorkItem is a no-op).
    timerWorkItem?.cancel()

    // Propagate timeout error
    if let errMsg = box.error, errMsg.hasPrefix("TIMEOUT:") {
        responsePtr.pointee.error_message = duplicateCString(errMsg)
        return responsePtr
    }

    if let text = box.text {
        responsePtr.pointee.text = duplicateCString(text)
        responsePtr.pointee.success = 1
    } else {
        let rawErr = box.error ?? "Unknown SpeechAnalyzer error."
        let prefixedErr = rawErr.hasPrefix("TIMEOUT:") || rawErr.hasPrefix("PERM_DENIED:") || rawErr.hasPrefix("AUTH_TIMEOUT:")
            ? rawErr
            : rawErr.hasPrefix("ENGINE:") ? rawErr : "ENGINE: \(rawErr)"
        responsePtr.pointee.error_message = duplicateCString(prefixedErr)
    }

    return responsePtr
}

// MARK: - Legacy path: SFSpeechRecognizer (macOS 10.15-25)

/// Legacy implementation kept verbatim from the original code.
/// Used on macOS 10.15 through 25 where SpeechAnalyzer is not available.
@available(macOS 10.15, *)
private func transcribeImplLegacy(
    samples: UnsafePointer<Float>,
    sampleCount: Int,
    sampleRate: Double,
    localeBcp47: UnsafePointer<CChar>,
    contextualStrings: UnsafePointer<UnsafePointer<CChar>?>?,
    contextualCount: Int,
    requireOnDevice: Int32,
    timeoutMs: Int32,
    onPartial: ((String) -> Void)?
) -> UnsafeMutablePointer<AppleSpeechResponse> {
    print("[apple_speech] Engine: SFSpeechRecognizer (legacy, macOS <26)")

    let responsePtr = ResponsePointer.allocate(capacity: 1)
    responsePtr.initialize(to: AppleSpeechResponse(text: nil, success: 0, error_message: nil))

    let localeStr = String(cString: localeBcp47)
    let locale = Locale(identifier: localeStr)

    guard let recognizer = SFSpeechRecognizer(locale: locale) else {
        responsePtr.pointee.error_message = duplicateCString(
            "SFSpeechRecognizer could not be created for locale: \(localeStr)"
        )
        return responsePtr
    }

    // Fast-fail if the recognizer reports it is not available (e.g. locale not
    // downloaded, server unreachable for network-only locale on first use).
    // NOTE: isAvailable is KVO-observable and may flip to true moments later when
    // the on-device model finishes loading, but for an immediate call we trust the
    // current value. A `false` here usually means the locale model is not installed;
    // the GCD timeout below is the safety net for the transient-not-ready case.
    if !recognizer.isAvailable {
        print("[apple_speech] WARNING: recognizer.isAvailable == false for locale '\(localeStr)'. Proceeding anyway — may time out.")
    }

    // Build contextual strings array
    var contextualArray: [String] = []
    if let ptr = contextualStrings, contextualCount > 0 {
        for i in 0..<contextualCount {
            if let cstr = ptr[i] {
                contextualArray.append(String(cString: cstr))
            }
        }
    }

    // Thread-safe container to pass results from async callback back to calling thread
    final class ResultBox: @unchecked Sendable {
        var text: String?
        var error: String?
    }
    let box = ResultBox()
    let semaphore = DispatchSemaphore(value: 0)

    // Request authorization first (blocking, using semaphore pattern).
    // This should not show a dialog on subsequent calls once permission is granted.
    // Use a 5-second timeout so automated/headless contexts (no one to click the dialog)
    // never hang indefinitely.
    let authSemaphore = DispatchSemaphore(value: 0)
    var authStatus: SFSpeechRecognizerAuthorizationStatus = SFSpeechRecognizer.authorizationStatus()

    if authStatus == .notDetermined {
        // Only call requestAuthorization when truly undetermined; this may show a dialog.
        SFSpeechRecognizer.requestAuthorization { status in
            authStatus = status
            authSemaphore.signal()
        }
        let authDeadline = DispatchTime.now() + .seconds(5)
        if authSemaphore.wait(timeout: authDeadline) == .timedOut {
            responsePtr.pointee.error_message = duplicateCString(
                "AUTH_TIMEOUT: Speech recognition authorization dialog timed out (5s). Please grant permission in System Settings → Privacy & Security → Speech Recognition."
            )
            return responsePtr
        }
    }

    switch authStatus {
    case .authorized:
        break // proceed
    case .denied:
        responsePtr.pointee.error_message = duplicateCString(
            "PERM_DENIED: Speech recognition authorization denied. Please enable in System Settings → Privacy & Security → Speech Recognition."
        )
        return responsePtr
    case .restricted:
        responsePtr.pointee.error_message = duplicateCString(
            "PERM_DENIED: Speech recognition is restricted on this device."
        )
        return responsePtr
    case .notDetermined:
        responsePtr.pointee.error_message = duplicateCString(
            "PERM_DENIED: Speech recognition authorization not determined after request."
        )
        return responsePtr
    @unknown default:
        responsePtr.pointee.error_message = duplicateCString(
            "PERM_DENIED: Unknown speech recognition authorization status."
        )
        return responsePtr
    }

    // Build AVAudioFormat for the input PCM buffer
    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: sampleRate,
        channels: 1,
        interleaved: false
    ) else {
        responsePtr.pointee.error_message = duplicateCString(
            "Failed to create AVAudioFormat for sample rate \(sampleRate)."
        )
        return responsePtr
    }

    guard let pcmBuffer = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(sampleCount)
    ) else {
        responsePtr.pointee.error_message = duplicateCString(
            "Failed to allocate AVAudioPCMBuffer."
        )
        return responsePtr
    }

    pcmBuffer.frameLength = AVAudioFrameCount(sampleCount)

    // Copy samples into the buffer
    if let channelData = pcmBuffer.floatChannelData {
        channelData[0].update(from: samples, count: sampleCount)
    }

    // Configure the recognition request; enable partial results when a callback is provided
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = onPartial != nil

    // Dictation hint: tells the recognizer this is free-form speech input, not a
    // search query or confirmation. Available since macOS 10.15.
    request.taskHint = .dictation

    // Native punctuation is intentionally DISABLED for incremental-paste safety.
    // When addsPunctuation = true, Apple's partial result stream retroactively
    // INSERTS punctuation into earlier positions ("你好世界" → "你好，世界").
    // The cumulative partial then no longer extends as a clean prefix, breaking
    // compute_delta() in clipboard.rs which relies on prefix-monotonic growth
    // for safe incremental paste to the target app's cursor.
    // We rely on the downstream punc_zh.rs post-processing for Chinese punctuation.
    //
    // To re-enable, we would need a non-trivial fuzzy-prefix matcher in
    // compute_delta plus a backspace-and-rewrite fallback for diverged finals.
    // if #available(macOS 13.0, *) {
    //     request.addsPunctuation = true
    // }

    // requiresOnDeviceRecognition is correctly set on the request (not the recognizer).
    // Setting it on the recognizer object itself was available in older SDKs but the
    // request-level flag is the right place since macOS 13 and is what Apple recommends.
    if requireOnDevice != 0 {
        request.requiresOnDeviceRecognition = true
    }
    if !contextualArray.isEmpty {
        request.contextualStrings = contextualArray
    }

    // Feed the PCM buffer and signal end of audio
    request.append(pcmBuffer)
    request.endAudio()

    // Start recognition task; keep a reference so the GCD timer can cancel it.
    var recognitionTask: SFSpeechRecognitionTask?
    recognitionTask = recognizer.recognitionTask(with: request) { result, error in
        if let result = result {
            if result.isFinal {
                box.text = result.bestTranscription.formattedString
                semaphore.signal()
            } else if let cb = onPartial {
                // Deliver partial result to the caller; the closure is called synchronously
                // here (on a Speech framework dispatch queue) and returns before we continue.
                cb(result.bestTranscription.formattedString)
            }
        } else if let error = error {
            box.error = error.localizedDescription
            semaphore.signal()
        }
    }

    // GCD timer guard: independently force-cancels the recognition task and signals
    // the semaphore after timeoutMs.  This ensures the semaphore is ALWAYS signaled
    // even when the SFSpeech completion handler never fires (e.g. permission not granted,
    // on-device model missing, or framework deadlock).
    var timerWorkItem: DispatchWorkItem?
    if timeoutMs > 0 {
        let workItem = DispatchWorkItem {
            recognitionTask?.cancel()
            recognitionTask = nil
            if box.text == nil && box.error == nil {
                box.error = "TIMEOUT: Apple Speech timed out after \(timeoutMs)ms. The recognizer may be unavailable or waiting for first-use initialization."
            }
            semaphore.signal()
        }
        timerWorkItem = workItem
        DispatchQueue.global(qos: .userInitiated).asyncAfter(
            deadline: .now() + .milliseconds(Int(timeoutMs)),
            execute: workItem
        )
    }

    // Wait — either the recognition callback or the GCD timer will signal us.
    semaphore.wait()

    // Cancel the timer if recognition finished first (race-safe: cancelling an already
    // executed DispatchWorkItem is a no-op).
    timerWorkItem?.cancel()

    // Propagate timeout error set by the timer work item
    if let errMsg = box.error, errMsg.hasPrefix("TIMEOUT:") {
        responsePtr.pointee.error_message = duplicateCString(errMsg)
        return responsePtr
    }

    if let text = box.text {
        responsePtr.pointee.text = duplicateCString(text)
        responsePtr.pointee.success = 1
    } else {
        let rawErr = box.error ?? "Unknown speech recognition error."
        // Ensure engine-level errors are prefixed so Rust can classify them.
        let prefixedErr = rawErr.hasPrefix("TIMEOUT:") || rawErr.hasPrefix("PERM_DENIED:") || rawErr.hasPrefix("AUTH_TIMEOUT:")
            ? rawErr
            : "ENGINE: \(rawErr)"
        responsePtr.pointee.error_message = duplicateCString(prefixedErr)
    }

    return responsePtr
}

// MARK: - Auth status query (C-callable, no dialog)

@_cdecl("apple_speech_get_auth_status")
public func appleSpeechGetAuthStatus() -> Int32 {
    guard #available(macOS 10.15, *) else { return -1 }
    let status = SFSpeechRecognizer.authorizationStatus()
    switch status {
    case .authorized:      return 3
    case .denied:          return 2
    case .restricted:      return 1
    case .notDetermined:   return 0
    @unknown default:      return -2
    }
}

// MARK: - Public C-callable transcription entry points

@_cdecl("transcribe_pcm_f32_apple_speech")
public func transcribePcmF32AppleSpeech(
    _ samples: UnsafePointer<Float>,
    _ sampleCount: Int,
    _ sampleRate: Double,
    _ localeBcp47: UnsafePointer<CChar>,
    _ contextualStrings: UnsafePointer<UnsafePointer<CChar>?>?,
    _ contextualCount: Int,
    _ requireOnDevice: Int32,
    _ timeoutMs: Int32
) -> UnsafeMutablePointer<AppleSpeechResponse> {
    guard #available(macOS 10.15, *) else {
        let responsePtr = ResponsePointer.allocate(capacity: 1)
        responsePtr.initialize(to: AppleSpeechResponse(text: nil, success: 0, error_message: nil))
        responsePtr.pointee.error_message = duplicateCString(
            "Apple Speech requires macOS 10.15 or newer."
        )
        return responsePtr
    }
    return transcribeImpl(
        samples: samples,
        sampleCount: sampleCount,
        sampleRate: sampleRate,
        localeBcp47: localeBcp47,
        contextualStrings: contextualStrings,
        contextualCount: contextualCount,
        requireOnDevice: requireOnDevice,
        timeoutMs: timeoutMs,
        onPartial: nil
    )
}

@_cdecl("transcribe_pcm_f32_apple_speech_with_partials")
public func transcribePcmF32AppleSpeechWithPartials(
    _ samples: UnsafePointer<Float>,
    _ sampleCount: Int,
    _ sampleRate: Double,
    _ localeBcp47: UnsafePointer<CChar>,
    _ contextualStrings: UnsafePointer<UnsafePointer<CChar>?>?,
    _ contextualCount: Int,
    _ requireOnDevice: Int32,
    _ timeoutMs: Int32,
    _ partialCb: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?,
    _ userData: UnsafeMutableRawPointer?
) -> UnsafeMutablePointer<AppleSpeechResponse> {
    guard #available(macOS 10.15, *) else {
        let responsePtr = ResponsePointer.allocate(capacity: 1)
        responsePtr.initialize(to: AppleSpeechResponse(text: nil, success: 0, error_message: nil))
        responsePtr.pointee.error_message = duplicateCString(
            "Apple Speech requires macOS 10.15 or newer."
        )
        return responsePtr
    }

    // Build an optional Swift closure that wraps the C callback.
    // The String's UTF-8 bytes are passed as a transient C string; the callback
    // must copy any data it needs before returning.
    let onPartial: ((String) -> Void)?
    if let cb = partialCb {
        onPartial = { text in
            text.withCString { cStr in
                cb(cStr, userData)
            }
        }
    } else {
        onPartial = nil
    }

    return transcribeImpl(
        samples: samples,
        sampleCount: sampleCount,
        sampleRate: sampleRate,
        localeBcp47: localeBcp47,
        contextualStrings: contextualStrings,
        contextualCount: contextualCount,
        requireOnDevice: requireOnDevice,
        timeoutMs: timeoutMs,
        onPartial: onPartial
    )
}

// MARK: - Memory management

@_cdecl("free_apple_speech_response")
public func freeAppleSpeechResponse(_ response: UnsafeMutablePointer<AppleSpeechResponse>?) {
    guard let response = response else { return }

    if let textStr = response.pointee.text {
        free(UnsafeMutablePointer(mutating: textStr))
    }

    if let errorStr = response.pointee.error_message {
        free(UnsafeMutablePointer(mutating: errorStr))
    }

    response.deallocate()
}
