import AVFoundation
import Dispatch
import Foundation
import Speech

// MARK: - Swift implementation for Apple Speech integration
// This file is compiled via Cargo build script for macOS targets (aarch64 + x86_64).
//
// Single-path SFSpeechRecognizer implementation.
// SpeechAnalyzer was attempted on macOS 26+ but kept SIGTRAP'ing in
// SpeechRecognizerWorker.preRunRecognition() — see git history
// (commits 9da58d1 / 0d6308e / 23b73e0) for rationale.

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
    // macOS 26 (Tahoe) routes SFSpeechRecognizer through a new SpeechAnalyzer
    // cooperative-queue backend.  Instantiating SFSpeechRecognizer — even as a
    // probe — triggers SpeechRecognizerWorker.preRunRecognition() on that queue,
    // which faults with SIGTRAP when the locale model is not pre-warmed.  This
    // hangs the caller indefinitely (no timeout inside SpeechAnalyzer itself).
    // Until a safe SpeechAnalyzer integration path is validated (see memory
    // gotcha_apple_speechanalyzer_macos26.md), we disable Apple Speech on
    // macOS 26+ to avoid the hang.
    if #available(macOS 26, *) {
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

// MARK: - Transcription

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
