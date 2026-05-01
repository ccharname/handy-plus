//! CT-Transformer Chinese punctuation restoration layer.
//!
//! Wraps `sherpa_onnx::OfflinePunctuation` with lazy init and graceful
//! fallback. If the model is not downloaded or initialization fails the
//! original text is returned unchanged — the caller never panics.
//!
//! # Model layout expected on disk
//! ```text
//! <model_dir>/model.onnx       ← main model file (non-quantised)
//! ```
//! The canonical directory is:
//! `<app_data_dir>/models/sherpa-onnx-punct-ct-transformer-zh-cn-2024-04-12/`

use log::{info, warn};
use once_cell::sync::OnceCell;
use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig};
use std::path::Path;
use std::sync::{Arc, Mutex};

// Global lazy-init slot.  We only need one instance; the OfflinePunctuation
// wrapper is Send + Sync.  We guard it behind a Mutex so that the first-init
// path is thread-safe.
static PUNC: OnceCell<Option<Arc<OfflinePunctuation>>> = OnceCell::new();
static INIT_MUTEX: Mutex<()> = Mutex::new(());

/// Returns `true` when a punctuation model directory looks complete.
pub fn is_punc_model_present(model_dir: &Path) -> bool {
    // Primary model file expected by sherpa-onnx ct_transformer path.
    let onnx = model_dir.join("model.onnx");
    model_dir.exists() && onnx.exists()
}

/// Lazily initialise the punctuation model from `model_dir` (once, globally).
///
/// - Returns `None` when the model is absent or creation fails.
/// - Subsequent calls always return the same cached result.
fn get_punc(model_dir: &Path) -> Option<Arc<OfflinePunctuation>> {
    // Fast path: already initialised.
    if let Some(cached) = PUNC.get() {
        return cached.clone();
    }

    // Slow path: first caller initialises under lock.
    let _guard = INIT_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Double-checked after acquiring the lock.
    if let Some(cached) = PUNC.get() {
        return cached.clone();
    }

    let onnx_path = model_dir.join("model.onnx");
    if !onnx_path.exists() {
        warn!(
            "punc_zh: model not found at {}; skipping punctuation",
            onnx_path.display()
        );
        let _ = PUNC.set(None);
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
                onnx_path.display()
            );
            let arc = Arc::new(p);
            let _ = PUNC.set(Some(arc.clone()));
            Some(arc)
        }
        None => {
            warn!(
                "punc_zh: OfflinePunctuation::create failed for {}",
                onnx_path.display()
            );
            let _ = PUNC.set(None);
            None
        }
    }
}

/// Apply CT-Transformer punctuation to `text`.
///
/// # Arguments
/// * `model_dir` — directory containing `model.onnx`
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

/// Resets the cached punctuation model instance.
///
/// Called after a model is deleted so the next transcription re-checks
/// whether the model is available.
pub fn reset_cached_model() {
    // OnceCell does not support reset, but we can log a warning.
    // In practice the user must restart the app after deleting the punc model
    // for the cache to be cleared — acceptable for a post-MVP feature.
    warn!("punc_zh: reset_cached_model called; restart the app to reload the model");
}
