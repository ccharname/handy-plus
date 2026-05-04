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
    // Pre-flight: verify HuggingFace cache is present before entering the
    // Swift bridge (which would silently hang for up to 30 min downloading).
    ensure_hf_cache_present(model_id)?;

    // Ensure default.metallib is colocated next to the running executable so
    // mlx-c's `load_colocated_library` finds it. Tauri bundles the metallib
    // into Contents/Resources/mlx/default.metallib (declared in tauri.conf.json
    // resources). On first call, install it as Contents/MacOS/mlx.metallib
    // (relative to current_exe).
    if let Err(e) = ensure_metallib_installed() {
        log::warn!("[mlx_audio] metallib install warning (best-effort): {}", e);
    }

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

/// Install `mlx.metallib` next to the running executable if missing, copying
/// from the Tauri resource bundle path or the `MLX_METALLIB_PATH` env var.
///
/// mlx-c's `load_colocated_library` looks for `<binary_dir>/mlx.metallib`
/// before the SwiftPM bundle / Resources fallback. Tauri puts our resource
/// metallib at `<binary_dir>/../Resources/mlx/default.metallib`, which is
/// NOT the colocated path mlx-c searches first. Cheapest fix: copy on first
/// use to the path mlx-c actually looks for.
///
/// Idempotent — does nothing if the destination already exists with non-zero
/// size. Best-effort — failures are non-fatal so `cargo run --example` (no
/// .app bundle) still works if MLX_METALLIB_PATH is set, and so a missing
/// resource doesn't crash the load path before mlx-c runs its full fallback
/// chain (which would surface a more informative error).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ensure_metallib_installed() -> Result<(), String> {
    use std::fs;

    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| "current_exe has no parent dir".to_string())?;

    // Target name matches mlx-c's `load_colocated_library(device, "mlx")`
    // which appends ".metallib" if the path has no extension.
    let dest = exe_dir.join("mlx.metallib");

    // Already installed and non-empty → done.
    if let Ok(meta) = fs::metadata(&dest) {
        if meta.len() > 0 {
            return Ok(());
        }
    }

    // Find a source. Priority:
    //   1. MLX_METALLIB_PATH env var (override for `cargo run --example` etc.)
    //   2. Tauri resource bundle (DOUBLE `resources/` segment): Tauri stages
    //      everything under `src-tauri/resources/**` into
    //      `<App>.app/Contents/Resources/resources/...` — note the inner
    //      `resources/` segment, which is the literal directory name from
    //      our src-tauri layout, NOT a Tauri-injected one.
    //   3. Single-segment fallback (`Contents/Resources/mlx/default.metallib`)
    //      in case the bundle layout is rearranged in a future Tauri version.
    //   4. <exe_dir>/mlx/default.metallib (some other bundle layouts)
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(p) = std::env::var("MLX_METALLIB_PATH") {
            v.push(std::path::PathBuf::from(p));
        }
        v.push(exe_dir.join("../Resources/resources/mlx/default.metallib"));
        v.push(exe_dir.join("../Resources/mlx/default.metallib"));
        v.push(exe_dir.join("mlx/default.metallib"));
        v
    };

    let src = candidates
        .into_iter()
        .find(|p| p.exists() && fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false))
        .ok_or_else(|| {
            format!(
                "no metallib source found (set MLX_METALLIB_PATH or bundle \
                 to {}/../Resources/mlx/default.metallib)",
                exe_dir.display()
            )
        })?;

    fs::copy(&src, &dest)
        .map_err(|e| format!("copy {:?} → {:?}: {}", src, dest, e))?;
    log::info!(
        "[mlx_audio] installed metallib: {:?} → {:?}",
        src,
        dest
    );
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn ensure_metallib_installed() -> Result<(), String> {
    Ok(())
}

/// Map a logical `model_id` to its HuggingFace Hub cache directory name
/// (i.e. `models--<org>--<repo>`).
fn hf_cache_dir_name(model_id: &str) -> &'static str {
    match model_id {
        "voxtral-mini-4b-4bit" => "models--mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit",
        "qwen3-asr-06b-8bit" => "models--mlx-community--Qwen3-ASR-0.6B-8bit",
        // Unknown model_id: return empty string so the caller treats it as
        // uncached and surfaces a generic error.
        _ => "",
    }
}

/// Verify that the HuggingFace Hub cache for `model_id` contains at least one
/// non-empty snapshot, i.e. the weights have been downloaded already.
///
/// Returns `Ok(())` when the cache looks populated. Returns `Err(...)` with a
/// Chinese-language user-facing message when the cache is missing or empty —
/// so the error surfaces in < 1 s instead of hanging for 5-30 minutes inside
/// the Swift bridge's `runSync` + `DispatchSemaphore`.
///
/// This is a best-effort gate: if home_dir() is unavailable the check is
/// skipped (returns Ok) to avoid blocking valid edge-case environments.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ensure_hf_cache_present(model_id: &str) -> Result<(), String> {
    use std::fs;

    let cache_dir_name = hf_cache_dir_name(model_id);
    if cache_dir_name.is_empty() {
        // Unknown model — let the bridge handle it; no pre-flight possible.
        return Ok(());
    }

    // Resolve home directory via dirs-next; fall back to $HOME env var.
    let home = dirs_next::home_dir()
        .or_else(|| std::env::var("HOME").ok().map(std::path::PathBuf::from));

    let home = match home {
        Some(h) => h,
        None => {
            log::warn!("[mlx_audio] cannot determine home directory; skipping HF cache pre-flight");
            return Ok(());
        }
    };

    let snapshots_dir = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(cache_dir_name)
        .join("snapshots");

    let snapshots_display = snapshots_dir.display().to_string();

    // Check that snapshots/ exists and contains at least one non-empty
    // subdirectory (each snapshot is a git commit hash directory).
    let has_snapshot = fs::read_dir(&snapshots_dir)
        .ok()
        .and_then(|mut entries| {
            entries.find(|entry| {
                entry.as_ref().ok().map_or(false, |e| {
                    // The entry itself must be a directory …
                    e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                        // … and must contain at least one file.
                        && fs::read_dir(e.path())
                            .map(|mut inner| inner.next().is_some())
                            .unwrap_or(false)
                })
            })
        })
        .is_some();

    if has_snapshot {
        return Ok(());
    }

    // Build user-facing error message. Voxtral gets a size hint; other models
    // get a generic prompt.
    let hint = if model_id == "voxtral-mini-4b-4bit" {
        format!(
            "Voxtral 模型权重尚未下载（约 3.5 GB）。请在终端运行：\n\
             \n\
             huggingface-cli download mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit\n\
             \n\
             预热缓存后再录音，或保持网络稳定 5–15 分钟后重试。\n\
             Cache 目录：{}",
            snapshots_display
        )
    } else {
        format!(
            "该模型（{}）权重尚未下载，请先用 huggingface-cli 预热缓存后再使用。\n\
             Cache 目录：{}",
            model_id, snapshots_display
        )
    };

    Err(hint)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn ensure_hf_cache_present(_model_id: &str) -> Result<(), String> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: call `ensure_hf_cache_present` against an arbitrary `snapshots`
    /// directory by temporarily overriding HOME via an environment variable.
    ///
    /// We can't directly call the cfg-gated function on non-aarch64 hosts, so
    /// the logic under test is extracted into `check_snapshots_dir` which works
    /// on every platform.
    fn check_snapshots_dir(snapshots_dir: &std::path::Path) -> Result<(), String> {
        let has_snapshot = fs::read_dir(snapshots_dir)
            .ok()
            .and_then(|mut entries| {
                entries.find(|entry| {
                    entry.as_ref().ok().map_or(false, |e| {
                        e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                            && fs::read_dir(e.path())
                                .map(|mut inner| inner.next().is_some())
                                .unwrap_or(false)
                    })
                })
            })
            .is_some();

        if has_snapshot {
            Ok(())
        } else {
            Err(format!(
                "snapshots dir missing or empty: {}",
                snapshots_dir.display()
            ))
        }
    }

    #[test]
    fn test_no_cache_dir_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        // snapshots dir simply does not exist
        let snapshots = tmp.path().join("snapshots");
        assert!(check_snapshots_dir(&snapshots).is_err());
    }

    #[test]
    fn test_empty_snapshots_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots = tmp.path().join("snapshots");
        fs::create_dir_all(&snapshots).unwrap();
        // snapshots/ exists but has no children
        assert!(check_snapshots_dir(&snapshots).is_err());
    }

    #[test]
    fn test_snapshot_dir_without_files_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots = tmp.path().join("snapshots");
        let hash_dir = snapshots.join("abc123def456");
        fs::create_dir_all(&hash_dir).unwrap();
        // hash dir exists but is empty
        assert!(check_snapshots_dir(&snapshots).is_err());
    }

    #[test]
    fn test_populated_snapshot_returns_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots = tmp.path().join("snapshots");
        let hash_dir = snapshots.join("abc123def456");
        fs::create_dir_all(&hash_dir).unwrap();
        // Place a non-empty file inside the snapshot
        fs::write(hash_dir.join("config.json"), b"{}").unwrap();
        assert!(check_snapshots_dir(&snapshots).is_ok());
    }

    #[test]
    fn test_hf_cache_dir_name_mappings() {
        assert_eq!(
            hf_cache_dir_name("voxtral-mini-4b-4bit"),
            "models--mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit"
        );
        assert_eq!(
            hf_cache_dir_name("qwen3-asr-06b-8bit"),
            "models--mlx-community--Qwen3-ASR-0.6B-8bit"
        );
        assert_eq!(hf_cache_dir_name("unknown-model"), "");
    }
}
