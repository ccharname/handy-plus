#ifndef apple_speech_bridge_h
#define apple_speech_bridge_h

#include <stddef.h>  // size_t

// C-compatible function declarations for Apple Speech (SFSpeechRecognizer) bridge

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char* text;
    int success; // 0 for failure, 1 for success
    char* error_message; // Only valid when success = 0
} AppleSpeechResponse;

// Check if Apple Speech (SFSpeechRecognizer) is available on the device.
// Returns 1 if the framework is present and at least one on-device locale exists; 0 otherwise.
int is_apple_speech_available(void);

// Query current speech recognition authorization status WITHOUT triggering a dialog.
// Returns: 3 = authorized, 2 = denied, 1 = restricted, 0 = notDetermined, -1 = OS too old, -2 = unknown.
int apple_speech_get_auth_status(void);

// Transcribe PCM f32 audio samples using Apple's SFSpeechRecognizer.
// samples        – pointer to float32 samples, mono, interleaved
// sample_count   – number of samples
// sample_rate    – sample rate in Hz (e.g. 16000.0)
// locale_bcp47   – BCP-47 locale string (e.g. "en-US", "zh-CN")
// contextual_strings / contextual_count – hint words for better recognition
// require_on_device – if 1, set requiresOnDeviceRecognition = true
// timeout_ms     – max wait in ms; <= 0 means no timeout
AppleSpeechResponse* transcribe_pcm_f32_apple_speech(
    const float* samples,
    size_t sample_count,
    double sample_rate,
    const char* locale_bcp47,
    const char* const* contextual_strings,
    size_t contextual_count,
    int require_on_device,
    int timeout_ms
);

// Free memory allocated by an AppleSpeechResponse.
void free_apple_speech_response(AppleSpeechResponse* response);

// Callback type for streaming partial transcription results.
// partial_text is a UTF-8 C string valid only for the duration of the callback.
// user_data is the opaque pointer passed to transcribe_pcm_f32_apple_speech_with_partials.
typedef void (*PartialCallback)(const char* partial_text, void* user_data);

// Like transcribe_pcm_f32_apple_speech but calls partial_cb for each intermediate
// result before the final one.  partial_cb may be NULL (behaves like the plain variant).
AppleSpeechResponse* transcribe_pcm_f32_apple_speech_with_partials(
    const float* samples,
    size_t sample_count,
    double sample_rate,
    const char* locale_bcp47,
    const char* const* contextual_strings,
    size_t contextual_count,
    int require_on_device,
    int timeout_ms,
    PartialCallback partial_cb,
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif /* apple_speech_bridge_h */
