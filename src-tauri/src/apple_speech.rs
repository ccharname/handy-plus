use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int};

// Define the response structure matching the Swift/C side
#[repr(C)]
pub struct AppleSpeechResponse {
    pub text: *mut c_char,
    pub success: c_int,
    pub error_message: *mut c_char,
}

// Declarations for the Swift-exported C functions.
// We use #[link_name] to map Rust identifiers to the actual C symbol names.
extern "C" {
    #[link_name = "is_apple_speech_available"]
    fn ffi_is_apple_speech_available() -> c_int;

    #[link_name = "transcribe_pcm_f32_apple_speech"]
    fn ffi_transcribe_pcm_f32_apple_speech(
        samples: *const c_float,
        sample_count: usize,
        sample_rate: c_double,
        locale_bcp47: *const c_char,
        contextual_strings: *const *const c_char,
        contextual_count: usize,
        require_on_device: c_int,
        timeout_ms: c_int,
    ) -> *mut AppleSpeechResponse;

    #[link_name = "free_apple_speech_response"]
    fn ffi_free_apple_speech_response(response: *mut AppleSpeechResponse);
}

/// Returns true if Apple Speech (SFSpeechRecognizer) is available on this device.
/// Does not trigger an authorization dialog — only checks framework availability.
pub fn is_apple_speech_available() -> bool {
    unsafe { ffi_is_apple_speech_available() == 1 }
}

/// Transcribe a slice of 16-bit mono f32 PCM samples using Apple SFSpeechRecognizer.
///
/// # Arguments
/// * `samples`         – mono float32 PCM audio (typically 16 kHz)
/// * `sample_rate`     – sample rate in Hz
/// * `locale`          – BCP-47 locale string (e.g. "en-US", "zh-CN")
/// * `contextual`      – optional list of hint words for better recognition accuracy
/// * `require_on_device` – if true, forces on-device (private) recognition
/// * `timeout_ms`      – max wait in ms; 0 or negative means no timeout
pub fn transcribe(
    samples: &[f32],
    sample_rate: f64,
    locale: &str,
    contextual: &[String],
    require_on_device: bool,
    timeout_ms: i32,
) -> Result<String, String> {
    let locale_cstr = CString::new(locale).map_err(|e| e.to_string())?;

    // Build a Vec<CString> to own the data, then a Vec<*const c_char> to pass as array
    let contextual_cstrings: Vec<CString> = contextual
        .iter()
        .filter_map(|s| CString::new(s.as_str()).ok())
        .collect();
    let contextual_ptrs: Vec<*const c_char> =
        contextual_cstrings.iter().map(|s| s.as_ptr()).collect();

    let (ctx_ptr, ctx_count) = if contextual_ptrs.is_empty() {
        (std::ptr::null(), 0usize)
    } else {
        (contextual_ptrs.as_ptr(), contextual_ptrs.len())
    };

    let response_ptr = unsafe {
        ffi_transcribe_pcm_f32_apple_speech(
            samples.as_ptr(),
            samples.len(),
            sample_rate,
            locale_cstr.as_ptr(),
            ctx_ptr,
            ctx_count,
            if require_on_device { 1 } else { 0 },
            timeout_ms,
        )
    };

    if response_ptr.is_null() {
        return Err("Null response from Apple Speech".to_string());
    }

    let response = unsafe { &*response_ptr };

    let result = if response.success == 1 {
        if response.text.is_null() {
            Ok(String::new())
        } else {
            let c_str = unsafe { CStr::from_ptr(response.text) };
            Ok(c_str.to_string_lossy().into_owned())
        }
    } else {
        let error_c_str = if !response.error_message.is_null() {
            unsafe { CStr::from_ptr(response.error_message) }
        } else {
            CStr::from_bytes_with_nul(b"Unknown Apple Speech error\0").unwrap()
        };
        Err(error_c_str.to_string_lossy().into_owned())
    };

    // Free the Swift-allocated response
    unsafe { ffi_free_apple_speech_response(response_ptr) };

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability_check() {
        // Just verify the FFI call doesn't crash
        let available = is_apple_speech_available();
        println!("Apple Speech available: {}", available);
    }
}
