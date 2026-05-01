import Foundation

// Stub implementation when Apple Speech (SFSpeechRecognizer) is not available.
// This file is compiled via Cargo build script when the target platform is not macOS.

private typealias ResponsePointer = UnsafeMutablePointer<AppleSpeechResponse>

@_cdecl("is_apple_speech_available")
public func isAppleSpeechAvailable() -> Int32 {
    return 0
}

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
    let responsePtr = ResponsePointer.allocate(capacity: 1)
    responsePtr.initialize(to: AppleSpeechResponse(text: nil, success: 0, error_message: nil))

    let msg = "Apple Speech is not available in this build (macOS only)."
    responsePtr.pointee.error_message = strdup(msg)

    return responsePtr
}

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
