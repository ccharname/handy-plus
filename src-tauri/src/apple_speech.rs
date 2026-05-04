// apple_native preset was removed in M1; transcription dispatch no longer calls
// this module. Keeping it for is_apple_speech_available (model registry) and
// get_auth_status (commands/audio.rs permission UI). The FFI transcription
// helpers are kept for potential future re-enablement.
#![allow(dead_code)]
use log::info;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int, c_void};

// Define the response structure matching the Swift/C side
#[repr(C)]
pub struct AppleSpeechResponse {
    pub text: *mut c_char,
    pub success: c_int,
    pub error_message: *mut c_char,
}

/// C callback type for streaming partial results from the Swift engine.
/// Called on a Speech framework dispatch queue; must return quickly.
/// `partial_text` is a transient UTF-8 C string valid only for the callback duration.
pub type PartialCallback = unsafe extern "C" fn(*const c_char, *mut c_void);

/// Parsed classification of an error string returned by the Swift layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppleSpeechError {
    /// Speech recognition permission denied or restricted.
    PermissionDenied(String),
    /// Authorization dialog timed out (headless / no user present).
    AuthTimeout(String),
    /// Recognition timed out (GCD timer fired before completion handler).
    Timeout(String),
    /// Framework/engine error (all other errors).
    Engine(String),
}

impl std::fmt::Display for AppleSpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppleSpeechError::PermissionDenied(msg) => write!(
                f,
                "Apple Speech permission not granted: {}. Please authorize in System Settings → Privacy & Security → Speech Recognition.",
                msg
            ),
            AppleSpeechError::AuthTimeout(msg) => write!(
                f,
                "Apple Speech authorization timed out: {}",
                msg
            ),
            AppleSpeechError::Timeout(msg) => write!(
                f,
                "Apple Speech timed out: {}. The recognizer may be stuck on first-use; please try again.",
                msg
            ),
            AppleSpeechError::Engine(msg) => write!(f, "Apple Speech engine error: {}", msg),
        }
    }
}

/// Parse the prefixed error strings returned by the Swift layer into a typed error.
pub fn parse_apple_speech_error(raw: &str) -> AppleSpeechError {
    if let Some(rest) = raw.strip_prefix("PERM_DENIED: ") {
        AppleSpeechError::PermissionDenied(rest.to_owned())
    } else if let Some(rest) = raw.strip_prefix("AUTH_TIMEOUT: ") {
        AppleSpeechError::AuthTimeout(rest.to_owned())
    } else if let Some(rest) = raw.strip_prefix("TIMEOUT: ") {
        AppleSpeechError::Timeout(rest.to_owned())
    } else if let Some(rest) = raw.strip_prefix("ENGINE: ") {
        AppleSpeechError::Engine(rest.to_owned())
    } else {
        // Legacy / unprefixed error — treat as engine error.
        AppleSpeechError::Engine(raw.to_owned())
    }
}

// Declarations for the Swift-exported C functions.
// We use #[link_name] to map Rust identifiers to the actual C symbol names.
extern "C" {
    #[link_name = "is_apple_speech_available"]
    fn ffi_is_apple_speech_available() -> c_int;

    #[link_name = "apple_speech_get_auth_status"]
    fn ffi_apple_speech_get_auth_status() -> c_int;

    #[link_name = "transcribe_pcm_f32_apple_speech_with_partials"]
    fn ffi_transcribe_pcm_f32_apple_speech_with_partials(
        samples: *const c_float,
        sample_count: usize,
        sample_rate: c_double,
        locale_bcp47: *const c_char,
        contextual_strings: *const *const c_char,
        contextual_count: usize,
        require_on_device: c_int,
        timeout_ms: c_int,
        partial_cb: Option<PartialCallback>,
        user_data: *mut c_void,
    ) -> *mut AppleSpeechResponse;

    #[link_name = "free_apple_speech_response"]
    fn ffi_free_apple_speech_response(response: *mut AppleSpeechResponse);
}

/// Returns true if Apple Speech (SFSpeechRecognizer) is available on this device.
/// Does not trigger an authorization dialog — only checks framework availability.
pub fn is_apple_speech_available() -> bool {
    unsafe { ffi_is_apple_speech_available() == 1 }
}

/// Speech recognition authorization status (mirrors SFSpeechRecognizerAuthorizationStatus).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum SpeechAuthStatus {
    Authorized,
    Denied,
    Restricted,
    NotDetermined,
    Unsupported,
}

/// Query the current speech recognition authorization status WITHOUT showing a dialog.
pub fn get_auth_status() -> SpeechAuthStatus {
    let raw = unsafe { ffi_apple_speech_get_auth_status() };
    match raw {
        3 => SpeechAuthStatus::Authorized,
        2 => SpeechAuthStatus::Denied,
        1 => SpeechAuthStatus::Restricted,
        0 => SpeechAuthStatus::NotDetermined,
        _ => SpeechAuthStatus::Unsupported,
    }
}

/// Sanitise a raw list of custom words before passing them to Apple's
/// `contextualStrings` API.
///
/// Apple's guidance: keep the list to ≤100 entries, each ≤10 chars, for best
/// performance. We apply a softer 80-char per-entry cap (brand names like
/// "GitHub Copilot" are 14 chars and should be kept; only obvious garbage /
/// pasted paragraphs are dropped).  Entries that are entirely ASCII punctuation
/// are dropped. Case-insensitive dedup is applied, and each entry is
/// whitespace-trimmed.
///
/// The result is logged so the count delta is visible in diagnostics (mirrors
/// the FunASR-Nano hotword logging pattern in the Qwen3 arm).
pub fn sanitise_contextual_strings(raw: &[String]) -> Vec<String> {
    const MAX_ENTRIES: usize = 100;
    const MAX_CHARS: usize = 80;

    let original_count = raw.len();
    let mut seen_lower: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result: Vec<String> = Vec::with_capacity(raw.len().min(MAX_ENTRIES));

    for entry in raw {
        let trimmed = entry.trim().to_string();

        // Skip empty
        if trimmed.is_empty() {
            continue;
        }

        // Skip entries that are only ASCII punctuation / symbols
        if trimmed
            .chars()
            .all(|c| c.is_ascii_punctuation() || c.is_ascii_whitespace())
        {
            continue;
        }

        // Skip entries that are too long (likely pasted paragraphs)
        if trimmed.chars().count() > MAX_CHARS {
            continue;
        }

        // Case-insensitive dedup
        let lower = trimmed.to_lowercase();
        if seen_lower.contains(&lower) {
            continue;
        }
        seen_lower.insert(lower);

        result.push(trimmed);

        if result.len() >= MAX_ENTRIES {
            break;
        }
    }

    info!(
        "Apple Speech contextualStrings: {} raw → {} after sanitise (cap={}, max_chars={})",
        original_count,
        result.len(),
        MAX_ENTRIES,
        MAX_CHARS
    );

    result
}

/// Transcribe PCM audio using Apple SFSpeechRecognizer, delivering partial results
/// to `on_partial` as they arrive.
///
/// # Arguments
/// * `on_partial` – called with each non-final transcription text. May be called 0 or more times.
pub fn transcribe_with_partials<F>(
    samples: &[f32],
    sample_rate: f64,
    locale: &str,
    contextual: &[String],
    require_on_device: bool,
    timeout_ms: i32,
    on_partial: F,
) -> Result<String, String>
where
    F: FnMut(&str) + Send,
{
    let locale_cstr = CString::new(locale).map_err(|e| e.to_string())?;

    // Sanitise contextual strings before passing through FFI
    let contextual_clean = sanitise_contextual_strings(contextual);

    let contextual_cstrings: Vec<CString> = contextual_clean
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

    // Box the closure so we can pass it through `*mut c_void`.
    // We use a double-box so the fat pointer is stored on the heap and we can
    // cast it to a thin `*mut c_void`.
    let boxed: Box<Box<dyn FnMut(&str) + Send>> = Box::new(Box::new(on_partial));
    let user_data: *mut c_void = Box::into_raw(boxed) as *mut c_void;

    // C shim: receives the transient `*const c_char` from Swift and forwards to
    // the Rust closure stored in `user_data`.
    unsafe extern "C" fn partial_shim(partial_text: *const c_char, user_data: *mut c_void) {
        if partial_text.is_null() || user_data.is_null() {
            return;
        }
        // Borrow the closure from user_data without consuming it (it may be called again).
        let cb = &mut *(user_data as *mut Box<dyn FnMut(&str) + Send>);
        let text = CStr::from_ptr(partial_text).to_string_lossy();
        cb(text.as_ref());
    }

    let response_ptr = unsafe {
        ffi_transcribe_pcm_f32_apple_speech_with_partials(
            samples.as_ptr(),
            samples.len(),
            sample_rate,
            locale_cstr.as_ptr(),
            ctx_ptr,
            ctx_count,
            if require_on_device { 1 } else { 0 },
            timeout_ms,
            Some(partial_shim),
            user_data,
        )
    };

    // Reclaim the boxed closure to drop it properly, regardless of outcome.
    // Safety: user_data was created by Box::into_raw above and Swift guarantees
    // it will not be used after the function returns.
    let _ = unsafe { Box::from_raw(user_data as *mut Box<dyn FnMut(&str) + Send>) };

    if response_ptr.is_null() {
        return Err("Null response from Apple Speech (with_partials)".to_string());
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
        let raw_err = if !response.error_message.is_null() {
            unsafe { CStr::from_ptr(response.error_message) }
                .to_string_lossy()
                .into_owned()
        } else {
            "Unknown Apple Speech error".to_string()
        };
        Err(parse_apple_speech_error(&raw_err).to_string())
    };

    unsafe { ffi_free_apple_speech_response(response_ptr) };

    result
}

// VERIFIED partial pipeline OK:
// - Swift partial_shim calls `cb(cStr, userData)` where cStr is a transient UTF-8
//   pointer valid for the callback's duration.
// - Rust's partial_shim copies it via CStr::from_ptr → to_string_lossy (handles
//   invalid UTF-8 gracefully via replacement chars; no boundary panic).
// - The double-box user_data is reclaimed after ffi_transcribe_pcm_f32_apple_speech_with_partials
//   returns, which Swift guarantees to be after all callbacks have fired.
// - The transcription-partial Tauri event carries `{ "text": "<str>" }` and the
//   overlay window listens for "transcription-partial". No race: final result is
//   only emitted once the Swift semaphore.signal() fires (isFinal == true),
//   which is mutually exclusive with further partial callbacks.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability_check() {
        // Just verify the FFI call doesn't crash
        let available = is_apple_speech_available();
        println!("Apple Speech available: {}", available);
    }

    #[test]
    fn test_sanitise_contextual_strings_basic() {
        let raw: Vec<String> = vec![
            "GitHub".to_string(),
            "github".to_string(),      // duplicate (case-insensitive) → dropped
            "  Copilot  ".to_string(), // trimmed → "Copilot"
            "...".to_string(),         // all ASCII punctuation → dropped
            "".to_string(),            // empty → dropped
        ];
        let result = sanitise_contextual_strings(&raw);
        assert_eq!(result, vec!["GitHub", "Copilot"]);
    }

    #[test]
    fn test_sanitise_contextual_strings_long_entry() {
        let long = "a".repeat(81);
        let raw = vec![long, "short".to_string()];
        let result = sanitise_contextual_strings(&raw);
        assert_eq!(result, vec!["short"]);
    }

    #[test]
    fn test_sanitise_contextual_strings_cap() {
        let raw: Vec<String> = (0..150).map(|i| format!("word{}", i)).collect();
        let result = sanitise_contextual_strings(&raw);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_sanitise_contextual_strings_brand_name() {
        // "GitHub Copilot" is 14 chars — must be kept (below 80-char cap)
        let raw = vec!["GitHub Copilot".to_string()];
        let result = sanitise_contextual_strings(&raw);
        assert_eq!(result, vec!["GitHub Copilot"]);
    }
}
