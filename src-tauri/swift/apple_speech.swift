import AVFoundation
import Dispatch
import Foundation
import Speech

// MARK: - Swift implementation for Apple Speech (SFSpeechRecognizer) integration
// This file is compiled via Cargo build script for macOS targets (aarch64 + x86_64).

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

// MARK: - Transcription (internal implementation)

/// Shared implementation for both the plain and with-partials variants.
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
    let authSemaphore = DispatchSemaphore(value: 0)
    var authStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined
    SFSpeechRecognizer.requestAuthorization { status in
        authStatus = status
        authSemaphore.signal()
    }
    authSemaphore.wait()

    switch authStatus {
    case .authorized:
        break // proceed
    case .denied:
        responsePtr.pointee.error_message = duplicateCString(
            "Speech recognition authorization denied. Please enable in System Preferences > Privacy > Speech Recognition."
        )
        return responsePtr
    case .restricted:
        responsePtr.pointee.error_message = duplicateCString(
            "Speech recognition is restricted on this device."
        )
        return responsePtr
    case .notDetermined:
        responsePtr.pointee.error_message = duplicateCString(
            "Speech recognition authorization not determined."
        )
        return responsePtr
    @unknown default:
        responsePtr.pointee.error_message = duplicateCString(
            "Unknown speech recognition authorization status."
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
    if requireOnDevice != 0 {
        request.requiresOnDeviceRecognition = true
    }
    if !contextualArray.isEmpty {
        request.contextualStrings = contextualArray
    }

    // Feed the PCM buffer and signal end of audio
    request.append(pcmBuffer)
    request.endAudio()

    // Start recognition task
    recognizer.recognitionTask(with: request) { result, error in
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

    // Wait with optional timeout
    let waitResult: DispatchTimeoutResult
    if timeoutMs <= 0 {
        semaphore.wait()
        waitResult = .success
    } else {
        let deadline = DispatchTime.now() + .milliseconds(Int(timeoutMs))
        waitResult = semaphore.wait(timeout: deadline)
    }

    if waitResult == .timedOut {
        responsePtr.pointee.error_message = duplicateCString(
            "Speech recognition timed out after \(timeoutMs)ms."
        )
        return responsePtr
    }

    if let text = box.text {
        responsePtr.pointee.text = duplicateCString(text)
        responsePtr.pointee.success = 1
    } else {
        responsePtr.pointee.error_message = duplicateCString(
            box.error ?? "Unknown speech recognition error."
        )
    }

    return responsePtr
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
