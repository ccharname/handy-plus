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
/// Lib build doesn't reference this (only `examples/test_mlx.rs` does), so allow
/// dead_code at the lib level.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(dead_code)]
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

    let c_model_id =
        CString::new(model_id).map_err(|e| format!("model_id contains NUL byte: {}", e))?;

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

/// Resolve the metallib source path from a prioritised list of candidates.
///
/// Returns the first candidate that exists and has non-zero size, or `Err`
/// when none is found.  Extracted as a pure function so it can be unit-tested
/// without a real bundle.
pub fn resolve_metallib_path(
    candidates: &[std::path::PathBuf],
) -> Result<std::path::PathBuf, String> {
    for p in candidates {
        if p.exists() {
            if let Ok(m) = std::fs::metadata(p) {
                if m.len() > 0 {
                    return Ok(p.clone());
                }
            }
        }
    }
    Err("no metallib source found among candidates".to_string())
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

    let src = resolve_metallib_path(&candidates).map_err(|_| {
        format!(
            "no metallib source found (set MLX_METALLIB_PATH or bundle \
             to {}/../Resources/mlx/default.metallib)",
            exe_dir.display()
        )
    })?;

    fs::copy(&src, &dest).map_err(|e| format!("copy {:?} → {:?}: {}", src, dest, e))?;
    log::info!("[mlx_audio] installed metallib: {:?} → {:?}", src, dest);
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

/// Key model files for Qwen3-ASR-MLX-8bit that must be present in every snapshot.
/// Checked as (filename, minimum_size_bytes, magic_prefix_hex).
///
/// Why size+magic instead of sha256:
///   - HF Hub does not expose stable sha256 for individual blobs without per-file
///     API queries (the `files-info` API is unstable / not in the public SDK).
///   - Model weights are large (100 MB – 2 GB per shard); hashing them at startup
///     costs 1–5 s on an SSD even when warm.
///   - Silent corruption is rare; the dominant failure mode is partial download
///     (zero-byte or truncated file), which size + magic catches perfectly.
///   - The `.metallib` magic and ONNX/safetensors magic are stable format markers.
///
/// Minimum sizes are conservative lower bounds (actual files are much larger).
const QWEN3_ASR_KEY_FILES: &[(&str, u64, &[u8])] = &[
    // config.json — plain JSON, ≥ 16 bytes, starts with `{`
    ("config.json", 16, b"{"),
    // tokenizer_config.json — Qwen3-ASR-0.6B-8bit uses BPE (separate vocab.json
    // + merges.txt) rather than a unified tokenizer.json, so we check the
    // tokenizer config file instead. JSON, ≥ 64 bytes. Verified against actual
    // mlx-community/Qwen3-ASR-0.6B-8bit snapshot 89e96d92ba34aca20b3e29fb10cc.
    ("tokenizer_config.json", 64, b"{"),
];

/// Perform lightweight integrity check on a snapshot directory:
/// verify that key files exist, exceed minimum sizes, and start with
/// expected magic bytes.
///
/// Returns `Ok(())` on pass, `Err(description)` on the first failing file.
pub fn check_snapshot_integrity(
    snapshot_dir: &std::path::Path,
    key_files: &[(&str, u64, &[u8])],
) -> Result<(), String> {
    use std::io::Read;
    for &(filename, min_size, magic) in key_files {
        let path = snapshot_dir.join(filename);
        let meta = std::fs::metadata(&path)
            .map_err(|e| format!("key file missing: {} ({})", path.display(), e))?;
        if meta.len() < min_size {
            return Err(format!(
                "key file too small: {} ({} bytes < {} minimum)",
                path.display(),
                meta.len(),
                min_size
            ));
        }
        if !magic.is_empty() {
            let mut buf = vec![0u8; magic.len()];
            let mut f = std::fs::File::open(&path)
                .map_err(|e| format!("cannot open key file {}: {}", path.display(), e))?;
            let n = f
                .read(&mut buf)
                .map_err(|e| format!("cannot read key file {}: {}", path.display(), e))?;
            if n < magic.len() || &buf[..n] != magic {
                return Err(format!(
                    "key file magic mismatch: {} (expected {:?}, got {:?})",
                    path.display(),
                    magic,
                    &buf[..n]
                ));
            }
        }
    }
    Ok(())
}

/// Verify that the HuggingFace Hub cache for `model_id` contains at least one
/// non-empty snapshot with valid key files (existence + size + magic bytes).
///
/// Returns `Ok(())` when the cache looks populated and intact.
/// Returns `Err(...)` with a Chinese-language user-facing message when the
/// cache is missing, empty, or corrupted — so the error surfaces in < 1 s
/// instead of hanging for 5-30 minutes inside the Swift bridge.
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
    let home =
        dirs_next::home_dir().or_else(|| std::env::var("HOME").ok().map(std::path::PathBuf::from));

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

    // Select the key-files manifest for this model_id.
    let key_files: &[(&str, u64, &[u8])] = if model_id == "qwen3-asr-06b-8bit" {
        QWEN3_ASR_KEY_FILES
    } else {
        // For other models, just check snapshot directory existence and non-empty.
        &[]
    };

    // Find a snapshot directory that passes the integrity check.
    let valid_snapshot = fs::read_dir(&snapshots_dir)
        .ok()
        .and_then(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
                .find(|e| {
                    let snapshot_dir = e.path();
                    // Must have at least one file.
                    let non_empty = fs::read_dir(&snapshot_dir)
                        .map(|mut inner| inner.next().is_some())
                        .unwrap_or(false);
                    if !non_empty {
                        return false;
                    }
                    // Run integrity check if key files are defined.
                    if !key_files.is_empty() {
                        match check_snapshot_integrity(&snapshot_dir, key_files) {
                            Ok(()) => true,
                            Err(e) => {
                                log::warn!(
                                    "[mlx_audio] snapshot integrity check failed for {:?}: {}",
                                    snapshot_dir,
                                    e
                                );
                                false
                            }
                        }
                    } else {
                        true
                    }
                })
        })
        .is_some();

    if valid_snapshot {
        return Ok(());
    }

    // Build user-facing error message.
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
    } else if model_id == "qwen3-asr-06b-8bit" {
        format!(
            "Qwen3-ASR 模型权重未下载或文件损坏。请在终端运行：\n\
             \n\
             huggingface-cli download mlx-community/Qwen3-ASR-0.6B-8bit\n\
             \n\
             若已下载但仍报此错，请删除缓存目录后重新下载：\n\
             rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit\n\
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

    // ── Legacy snapshot-dir presence tests ──────────────────────────────────

    /// Helper: reproduce the snapshot-presence check as a platform-independent
    /// pure function (the cfg-gated `ensure_hf_cache_present` is only compiled
    /// on aarch64).
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
        let snapshots = tmp.path().join("snapshots");
        assert!(check_snapshots_dir(&snapshots).is_err());
    }

    #[test]
    fn test_empty_snapshots_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots = tmp.path().join("snapshots");
        fs::create_dir_all(&snapshots).unwrap();
        assert!(check_snapshots_dir(&snapshots).is_err());
    }

    #[test]
    fn test_snapshot_dir_without_files_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots = tmp.path().join("snapshots");
        let hash_dir = snapshots.join("abc123def456");
        fs::create_dir_all(&hash_dir).unwrap();
        assert!(check_snapshots_dir(&snapshots).is_err());
    }

    #[test]
    fn test_populated_snapshot_returns_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots = tmp.path().join("snapshots");
        let hash_dir = snapshots.join("abc123def456");
        fs::create_dir_all(&hash_dir).unwrap();
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

    // ── HF cache integrity tests (M3) ────────────────────────────────────────

    /// File missing → Err
    #[test]
    fn hf_cache_integrity_missing_file_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let key_files: &[(&str, u64, &[u8])] = &[("config.json", 1, b"{")];
        let result = check_snapshot_integrity(tmp.path(), key_files);
        assert!(result.is_err(), "missing key file must return Err");
        assert!(result.unwrap_err().contains("missing"));
    }

    /// File present but too small → Err
    #[test]
    fn hf_cache_integrity_too_small_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), b"x").unwrap(); // 1 byte
        let key_files: &[(&str, u64, &[u8])] = &[("config.json", 100, b"x")]; // min=100
        let result = check_snapshot_integrity(tmp.path(), key_files);
        assert!(result.is_err(), "too-small file must return Err");
        assert!(result.unwrap_err().contains("too small"));
    }

    /// File present and large enough but wrong magic → Err
    #[test]
    fn hf_cache_integrity_wrong_magic_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        // Write a file starting with 'X' but we check for '{'
        let content: Vec<u8> = std::iter::repeat(b'X').take(200).collect();
        fs::write(tmp.path().join("config.json"), &content).unwrap();
        let key_files: &[(&str, u64, &[u8])] = &[("config.json", 16, b"{")];
        let result = check_snapshot_integrity(tmp.path(), key_files);
        assert!(result.is_err(), "wrong magic must return Err");
        assert!(result.unwrap_err().contains("magic"));
    }

    /// File present, large enough, correct magic → Ok
    #[test]
    fn hf_cache_integrity_valid_file_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut content = b"{ \"model_type\": \"qwen3_asr\" }".to_vec();
        content.extend(std::iter::repeat(b' ').take(200));
        fs::write(tmp.path().join("config.json"), &content).unwrap();
        let key_files: &[(&str, u64, &[u8])] = &[("config.json", 16, b"{")];
        let result = check_snapshot_integrity(tmp.path(), key_files);
        assert!(result.is_ok(), "valid file must pass: {:?}", result);
    }

    /// All key files present, sizes and magic correct → Ok
    #[test]
    fn hf_cache_integrity_all_files_valid_passes() {
        let tmp = tempfile::tempdir().unwrap();
        // config.json
        let cfg_content: Vec<u8> = {
            let mut v = b"{ \"model_type\": \"qwen3_asr\" }".to_vec();
            v.extend(std::iter::repeat(b' ').take(200));
            v
        };
        fs::write(tmp.path().join("config.json"), &cfg_content).unwrap();
        // tokenizer_config.json (Qwen3-ASR uses BPE; no unified tokenizer.json)
        let tok_content: Vec<u8> = {
            let mut v = b"{ \"tokenizer_class\": \"Qwen2Tokenizer\" }".to_vec();
            v.extend(std::iter::repeat(b' ').take(200));
            v
        };
        fs::write(tmp.path().join("tokenizer_config.json"), &tok_content).unwrap();
        let result = check_snapshot_integrity(tmp.path(), QWEN3_ASR_KEY_FILES);
        assert!(
            result.is_ok(),
            "all key files valid must pass: {:?}",
            result
        );
    }

    /// Empty magic slice → no magic check, only size check
    #[test]
    fn hf_cache_integrity_empty_magic_skips_magic_check() {
        let tmp = tempfile::tempdir().unwrap();
        let content: Vec<u8> = std::iter::repeat(b'Z').take(100).collect();
        fs::write(tmp.path().join("weights.bin"), &content).unwrap();
        // Empty magic: no magic check, only size
        let key_files: &[(&str, u64, &[u8])] = &[("weights.bin", 50, b"")];
        let result = check_snapshot_integrity(tmp.path(), key_files);
        assert!(
            result.is_ok(),
            "empty magic should skip magic check: {:?}",
            result
        );
    }

    // ── metallib path fallback tests (M3) ────────────────────────────────────

    /// Primary candidate exists → returned
    #[test]
    fn metallib_resolve_primary_exists_returns_primary() {
        let tmp = tempfile::tempdir().unwrap();
        let primary = tmp.path().join("primary.metallib");
        let fallback = tmp.path().join("fallback.metallib");
        fs::write(&primary, b"MTLB_CONTENT_PRIMARY").unwrap();
        fs::write(&fallback, b"MTLB_CONTENT_FALLBACK").unwrap();
        let candidates = vec![primary.clone(), fallback.clone()];
        let result = resolve_metallib_path(&candidates).unwrap();
        assert_eq!(result, primary, "primary exists → must return primary");
    }

    /// Primary missing, fallback exists → fallback returned
    #[test]
    fn metallib_resolve_fallback_when_primary_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let primary = tmp.path().join("nonexistent.metallib");
        let fallback = tmp.path().join("fallback.metallib");
        fs::write(&fallback, b"MTLB_CONTENT").unwrap();
        let candidates = vec![primary, fallback.clone()];
        let result = resolve_metallib_path(&candidates).unwrap();
        assert_eq!(result, fallback, "primary missing → must return fallback");
    }

    /// All candidates missing → Err
    #[test]
    fn metallib_resolve_all_missing_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.metallib");
        let b = tmp.path().join("b.metallib");
        let candidates = vec![a, b];
        let result = resolve_metallib_path(&candidates);
        assert!(result.is_err(), "all missing must return Err");
    }

    /// Zero-size file is skipped, non-zero next candidate is used
    #[test]
    fn metallib_resolve_skips_zero_size_file() {
        let tmp = tempfile::tempdir().unwrap();
        let zero = tmp.path().join("zero.metallib");
        let good = tmp.path().join("good.metallib");
        fs::write(&zero, b"").unwrap(); // zero bytes
        fs::write(&good, b"MTLB_NONZERO").unwrap();
        let candidates = vec![zero, good.clone()];
        let result = resolve_metallib_path(&candidates).unwrap();
        assert_eq!(result, good, "zero-size file must be skipped");
    }

    /// Empty candidate list → Err
    #[test]
    fn metallib_resolve_empty_candidates_returns_err() {
        let result = resolve_metallib_path(&[]);
        assert!(result.is_err());
    }
}
