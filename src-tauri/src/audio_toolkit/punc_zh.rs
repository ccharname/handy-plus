//! CT-Transformer Chinese punctuation restoration layer.
//!
//! Wraps `sherpa_onnx::OfflinePunctuation` with lazy init and graceful
//! fallback. If the model is not downloaded or initialization fails the
//! original text is returned unchanged — the caller never panics.
//!
//! # Model layout expected on disk
//! ```text
//! <model_dir>/model.int8.onnx       ← main model file (non-quantised)
//! ```
//! The canonical directory is:
//! `<app_data_dir>/models/sherpa-onnx-punct-ct-transformer-zh-cn-2024-04-12/`

use log::{info, warn};
use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig};
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Lifecycle state of the global punctuation model slot.
enum PuncState {
    /// No init attempt has been made yet (or the cache was explicitly reset).
    NotInited,
    /// Model was loaded successfully.
    InitedOk(Arc<OfflinePunctuation>),
    /// Last init attempt failed (model absent or `OfflinePunctuation::create` returned None).
    /// Stores the path that was tried so we can detect "model has since appeared".
    InitedFailed(std::path::PathBuf),
}

// SAFETY: OfflinePunctuation is Send + Sync (the underlying ONNX session is
// internally thread-safe).  We wrap it in Arc so clones are cheap.
unsafe impl Send for PuncState {}
unsafe impl Sync for PuncState {}

static PUNC: RwLock<PuncState> = RwLock::new(PuncState::NotInited);

/// Returns `true` when a punctuation model directory looks complete.
pub fn is_punc_model_present(model_dir: &Path) -> bool {
    // Primary model file expected by sherpa-onnx ct_transformer path.
    let onnx = model_dir.join("model.int8.onnx");
    model_dir.exists() && onnx.exists()
}

/// Lazily initialise the punctuation model from `model_dir`.
///
/// - Returns `Some(Arc<…>)` if the model is loaded (possibly on this call).
/// - Returns `None` when the model is absent or creation fails.
/// - Thread-safe: concurrent callers share a single `RwLock`; the slow-path
///   write section is double-checked to avoid redundant init.
/// - After `reset_cached_model()` a subsequent call will re-attempt init even
///   if a previous attempt failed — enabling "no-restart-required" model
///   download flow.
fn get_punc(model_dir: &Path) -> Option<Arc<OfflinePunctuation>> {
    let onnx_path = model_dir.join("model.int8.onnx");

    // ── Fast path (read lock) ────────────────────────────────────────────────
    {
        let guard = PUNC.read().unwrap_or_else(|e| e.into_inner());
        match &*guard {
            PuncState::InitedOk(arc) => return Some(arc.clone()),
            PuncState::InitedFailed(tried) => {
                // Only retry if the model file has since appeared.
                if !onnx_path.exists() || tried == &onnx_path {
                    // Model still absent and same path → no point retrying.
                    return None;
                }
                // Fall through to slow path — model file appeared since last try.
            }
            PuncState::NotInited => {
                // Fall through to slow path.
            }
        }
    }

    // ── Slow path (write lock, double-checked) ───────────────────────────────
    let mut guard = PUNC.write().unwrap_or_else(|e| e.into_inner());

    // Double-check: another thread may have inited between our read and write.
    match &*guard {
        PuncState::InitedOk(arc) => return Some(arc.clone()),
        PuncState::InitedFailed(tried) => {
            if !onnx_path.exists() || tried == &onnx_path {
                return None;
            }
        }
        PuncState::NotInited => {}
    }

    if !onnx_path.exists() {
        warn!(
            "punc_zh: model not found at {}; skipping punctuation",
            onnx_path.display()
        );
        *guard = PuncState::InitedFailed(onnx_path);
        return None;
    }

    let mut config = OfflinePunctuationConfig::default();
    config.model.ct_transformer = Some(onnx_path.to_string_lossy().to_string());
    // Keep thread-count at 1 — the model is invoked from the transcription
    // thread which already runs outside the main thread, and extra threads
    // inside the ONNX session add latency overhead for short strings.
    config.model.num_threads = 1;

    match OfflinePunctuation::create(&config) {
        Some(p) => {
            info!(
                "punc_zh: CT-Transformer model loaded from {}",
                model_dir.display()
            );
            let arc = Arc::new(p);
            *guard = PuncState::InitedOk(arc.clone());
            Some(arc)
        }
        None => {
            warn!(
                "punc_zh: OfflinePunctuation::create failed for {}",
                onnx_path.display()
            );
            *guard = PuncState::InitedFailed(onnx_path);
            None
        }
    }
}

/// Apply CT-Transformer punctuation to `text`.
///
/// # Arguments
/// * `model_dir` — directory containing `model.int8.onnx`
/// * `text`      — raw transcription text (may already have some punctuation)
///
/// # Returns
/// `Ok(punctuated)` on success, or `Err(reason)` when the model is missing
/// or inference fails.  The caller should fall back to returning `text`
/// unchanged on error.
pub fn add_punctuation(model_dir: &Path, text: &str) -> Result<String, String> {
    if text.is_empty() {
        return Ok(text.to_string());
    }

    let punc = get_punc(model_dir).ok_or_else(|| "punc_zh model not available".to_string())?;

    punc.add_punctuation(text)
        .ok_or_else(|| "punc_zh: add_punctuation returned None".to_string())
}

/// Reset the cached punctuation model state to `NotInited`.
///
/// The next call to `add_punctuation` (or `get_punc`) will re-attempt model
/// initialization from the model directory passed at that time.  Call this
/// after a model download/extraction completes so the app can use the model
/// immediately without requiring a restart.
pub fn reset_cached_model() {
    let mut guard = PUNC.write().unwrap_or_else(|e| e.into_inner());
    *guard = PuncState::NotInited;
    info!("punc_zh: model cache reset; next call will re-init");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    // Serialize tests that mutate the global PUNC static so they don't race
    // each other (Rust's test runner can run them concurrently by default).
    use std::sync::Mutex;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Helper: read the raw state variant name for assertions.
    fn state_variant() -> &'static str {
        let guard = PUNC.read().unwrap_or_else(|e| e.into_inner());
        match &*guard {
            PuncState::NotInited => "NotInited",
            PuncState::InitedOk(_) => "InitedOk",
            PuncState::InitedFailed(_) => "InitedFailed",
        }
    }

    #[test]
    fn reset_clears_cache() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 1. Drive the cache into InitedFailed via a nonexistent model dir.
        let bogus_dir = PathBuf::from("/nonexistent/punc_model_test_dir_xyz");
        let _ = get_punc(&bogus_dir); // sets InitedFailed
        assert_eq!(
            state_variant(),
            "InitedFailed",
            "expected InitedFailed after attempting a missing model"
        );

        // 2. reset_cached_model() should transition back to NotInited.
        reset_cached_model();
        assert_eq!(
            state_variant(),
            "NotInited",
            "expected NotInited after reset"
        );
    }

    #[test]
    fn starts_not_inited_or_failed() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // After a reset we must be NotInited.
        reset_cached_model();
        assert_eq!(state_variant(), "NotInited");
    }

    #[test]
    fn failed_path_not_retried_without_reset() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_cached_model();
        let bogus = PathBuf::from("/nonexistent/punc_model_stable_xyz");
        // First call → InitedFailed
        assert!(get_punc(&bogus).is_none());
        assert_eq!(
            state_variant(),
            "InitedFailed",
            "expected InitedFailed after attempting a missing model"
        );
        // Second call with same bogus path → still None, no infinite loop
        assert!(get_punc(&bogus).is_none());
        assert_eq!(
            state_variant(),
            "InitedFailed",
            "expected InitedFailed on repeated attempt with same path"
        );
        // Clean up for other tests
        reset_cached_model();
    }
}
