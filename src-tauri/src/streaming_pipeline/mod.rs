//! FD-006 M1 — Concurrent inference orchestrator for true streaming ASR.
//!
//! Architecture:
//!   recording thread → chunk channel → [`StreamingOrchestrator`] worker →
//!   [`ChunkPartial`] channel → caller (e.g. TranscriptionManager, M2)
//!
//! The worker pool is intentionally size-1 because mlx-audio-swift's Metal
//! inference serialises internally — spawning multiple workers would block on
//! Swift's dispatch queue, wasting threads. We apply a backpressure strategy
//! instead: if the queue fills up, we drop middle chunks to keep both the
//! oldest (partial continuity) and the newest (realtime feel).

pub mod orchestrator;

pub use orchestrator::{ChunkPartial, OrchestratorConfig, StreamingOrchestrator};

/// Trait abstraction for the inference backend.
///
/// Implementors receive raw 16 kHz f32 audio and emit cumulative partial
/// transcripts via `on_partial`, returning the final authoritative text.
///
/// The real implementation wraps `crate::mlx_audio::transcribe_streaming`.
/// Tests inject a `MockInferenceBackend` to avoid FFI calls.
pub trait InferenceBackend: Send + Sync + 'static {
    /// Run inference on `audio` (16 kHz f32 mono).
    ///
    /// `on_partial` is called zero or more times with **cumulative** partial
    /// text as tokens are decoded.  The last call carries the final
    /// authoritative result.
    ///
    /// Returns `Ok(final_text)` on success, `Err(reason)` on failure.
    fn run(
        &self,
        audio: &[f32],
        model_id: &str,
        on_partial: &mut dyn FnMut(&str),
    ) -> Result<String, String>;
}
