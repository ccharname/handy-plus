/// mlx_audio — Rust FFI wrapper for the MLXAudioBridge Swift static library.
///
/// Analogous to `apple_speech.rs`: exposes a thin safe API over the `@_cdecl`
/// exports compiled into `libmlx_audio.a` by `build.rs`.
///
/// Only compiled on macOS aarch64. All public functions return early with
/// `Err("MLX audio bridge not available")` on other platforms at compile time
/// (the cfg gates ensure the extern "C" block is never emitted there).
///
/// # Thread safety
/// `transcribe_file` blocks the calling thread via `DispatchSemaphore` inside
/// the Swift bridge until inference completes. Always call from a Cargo worker
/// thread, never from the main thread or a Tauri UI thread.
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
extern "C" {
    /// Returns the bridge + mlx-audio-swift version string as a heap-allocated
    /// C string. Caller must free with `mlx_audio_bridge_free_string`.
    #[link_name = "mlx_audio_bridge_version"]
    fn ffi_mlx_audio_bridge_version() -> *mut c_char;

    /// Free a C string returned by any `mlx_audio_bridge_*` or
    /// `mlx_audio_transcribe_file` function.
    #[link_name = "mlx_audio_bridge_free_string"]
    fn ffi_mlx_audio_bridge_free_string(ptr: *mut c_char);

    /// One-shot file transcription.
    /// Returns 0 on success; sets *out_text to a strdup'd result string.
    /// Returns non-zero on error; sets *out_error to a strdup'd error message.
    /// Caller must free both via `ffi_mlx_audio_bridge_free_string`.
    #[link_name = "mlx_audio_transcribe_file"]
    fn ffi_mlx_audio_transcribe_file(
        wav_path: *const c_char,
        model_id: *const c_char,
        out_text: *mut *mut c_char,
        out_error: *mut *mut c_char,
    ) -> i32;
}

/// Return the bridge version string (e.g. "handy-mlx-bridge/0.1.0 mlx-audio-swift/0.1.2").
/// Used as a startup smoke-test to confirm the library linked and Metal initialises.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn bridge_version() -> Result<String, String> {
    let ptr = unsafe { ffi_mlx_audio_bridge_version() };
    if ptr.is_null() {
        return Err("mlx_audio_bridge_version returned null".to_string());
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { ffi_mlx_audio_bridge_free_string(ptr) };
    Ok(s)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn bridge_version() -> Result<String, String> {
    Err("MLX audio bridge is only available on macOS Apple Silicon".to_string())
}

/// Transcribe a 16 kHz mono WAV file using the specified mlx-audio model.
///
/// `model_id` must be one of:
///   - `"voxtral-mini-4b-4bit"` → mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
///   - `"qwen3-asr-06b-8bit"`   → mlx-community/Qwen3-ASR-0.6B-8bit (reserved)
///
/// On first use the model weights are downloaded from HuggingFace Hub to
/// `~/.cache/huggingface/hub/` (~3.5 GB for Voxtral 4-bit). Subsequent calls
/// use the on-disk cache.
///
/// Returns the transcribed text on success, or a descriptive error string.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn transcribe_file(wav_path: &Path, model_id: &str) -> Result<String, String> {
    let c_path = CString::new(
        wav_path
            .to_str()
            .ok_or_else(|| "WAV path contains non-UTF-8 characters".to_string())?,
    )
    .map_err(|e| format!("WAV path contains NUL byte: {}", e))?;

    let c_model_id = CString::new(model_id)
        .map_err(|e| format!("model_id contains NUL byte: {}", e))?;

    let mut out_text: *mut c_char = std::ptr::null_mut();
    let mut out_error: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        ffi_mlx_audio_transcribe_file(
            c_path.as_ptr(),
            c_model_id.as_ptr(),
            &mut out_text,
            &mut out_error,
        )
    };

    if rc == 0 {
        // Success: read text and free the bridge-allocated string.
        let text = if out_text.is_null() {
            String::new()
        } else {
            let s = unsafe { CStr::from_ptr(out_text) }
                .to_string_lossy()
                .into_owned();
            unsafe { ffi_mlx_audio_bridge_free_string(out_text) };
            s
        };
        Ok(text)
    } else {
        // Error: read the error message and free it.
        let err_msg = if out_error.is_null() {
            format!("mlx_audio_transcribe_file returned error code {}", rc)
        } else {
            let s = unsafe { CStr::from_ptr(out_error) }
                .to_string_lossy()
                .into_owned();
            unsafe { ffi_mlx_audio_bridge_free_string(out_error) };
            s
        };
        Err(err_msg)
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn transcribe_file(_wav_path: &Path, _model_id: &str) -> Result<String, String> {
    Err("MLX audio bridge is only available on macOS Apple Silicon".to_string())
}
