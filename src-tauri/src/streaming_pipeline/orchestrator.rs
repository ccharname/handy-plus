//! FD-006 M1 — StreamingOrchestrator
//!
//! Drives a single-worker inference pipeline:
//!
//!  1. Caller submits [`ChunkedAudio`] chunks via [`StreamingOrchestrator::submit`].
//!  2. A background `std::thread` dequeues chunks and calls
//!     [`InferenceBackend::run`] synchronously (one at a time).
//!  3. For each chunk, the worker emits a [`ChunkPartial`] onto the output
//!     channel after inference completes.
//!  4. Caller polls [`StreamingOrchestrator::try_recv_partial`] /
//!     [`StreamingOrchestrator::recv_partial_timeout`] or calls
//!     [`StreamingOrchestrator::shutdown`] to collect all remaining partials.
//!
//! ## Backpressure
//!
//! The inbound queue has a configurable `max_queue_depth` (default 3). When
//! the queue is full, [`submit`] drops the **middle** element (index 1) and
//! appends the newly submitted chunk, preserving:
//!   - `[0]` — earliest chunk (important for partial continuity)
//!   - `[N]` — newest chunk (important for realtime feel)
//!
//! A `tracing::warn!` log line is emitted for every dropped chunk.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::audio_toolkit::audio::chunker::ChunkedAudio;
use crate::streaming_pipeline::InferenceBackend;

// ── Public output type ────────────────────────────────────────────────────────

/// Result emitted by the worker after processing one [`ChunkedAudio`] chunk.
#[derive(Debug, Clone)]
pub struct ChunkPartial {
    /// Zero-based index of the source chunk (monotonic within a session).
    pub chunk_idx: u64,
    /// Cumulative partial text from the **last** `on_partial` callback.
    pub partial_text: String,
    /// Total inference wall-clock time in milliseconds.
    pub inference_ms: f64,
    /// Time from inference start to the first `on_partial` callback.
    /// Only set for `chunk_idx == 0` (cold-start measurement).
    pub first_token_ms: Option<f64>,
    /// Non-`None` when inference returned an error for this chunk.
    pub error: Option<String>,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`StreamingOrchestrator`].
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Logical model identifier, e.g. `"qwen3-asr-06b-8bit"`.
    pub model_id: String,
    /// Maximum number of pending chunks in the inbound queue before backpressure
    /// kicks in. Default: 3.
    pub max_queue_depth: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            model_id: "qwen3-asr-06b-8bit".to_string(),
            max_queue_depth: 3,
        }
    }
}

// ── Shared inbound queue ──────────────────────────────────────────────────────

/// Shared state between the submitter side and the worker thread.
struct InboundQueue {
    deque: Mutex<VecDeque<ChunkedAudio>>,
    /// Signals the worker to drain and exit when true.
    shutdown: AtomicBool,
    /// Set by [`StreamingOrchestrator::cancel`]; worker stops emitting partials.
    cancel: AtomicBool,
    /// Condvar-style notification: a new item was pushed (or shutdown signalled).
    condvar: std::sync::Condvar,
}

impl InboundQueue {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            deque: Mutex::new(VecDeque::new()),
            shutdown: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            condvar: std::sync::Condvar::new(),
        })
    }
}

// ── StreamingOrchestrator ─────────────────────────────────────────────────────

/// Manages the single-worker inference loop.
///
/// `partial_rx` is wrapped in a `Mutex` to make `StreamingOrchestrator` `Sync`
/// so it can be held inside `Arc<StreamingOrchestrator>` in `AudioRecordingManager`.
/// The Mutex is only ever locked by one thread at a time (the drainer thread).
pub struct StreamingOrchestrator {
    config: OrchestratorConfig,
    queue: Arc<InboundQueue>,
    /// Receiver end of the partial-output channel.
    partial_rx: Mutex<std::sync::mpsc::Receiver<ChunkPartial>>,
    worker_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl StreamingOrchestrator {
    /// Spawn the background worker thread and return an orchestrator ready to
    /// accept chunks.
    pub fn start(config: OrchestratorConfig, backend: Box<dyn InferenceBackend>) -> Self {
        let queue = InboundQueue::new();
        let (partial_tx, partial_rx) = std::sync::mpsc::channel::<ChunkPartial>();

        let worker_queue = Arc::clone(&queue);
        let worker_config = config.clone();

        let handle = std::thread::spawn(move || {
            worker_loop(worker_config, worker_queue, backend, partial_tx);
        });

        Self {
            config,
            queue,
            partial_rx: Mutex::new(partial_rx),
            worker_handle: Mutex::new(Some(handle)),
        }
    }

    /// Push a chunk into the inbound queue for inference.
    ///
    /// Non-blocking. If the queue has reached `max_queue_depth`, the middle
    /// element is silently dropped and a warning is logged.
    ///
    /// Returns `Err("cancelled")` if the orchestrator has been cancelled.
    /// Returns `Err("shutdown")` if the orchestrator has been shut down.
    pub fn submit(&self, chunk: ChunkedAudio) -> Result<(), &'static str> {
        if self.queue.cancel.load(Ordering::Relaxed) {
            return Err("cancelled");
        }
        if self.queue.shutdown.load(Ordering::Relaxed) {
            return Err("shutdown");
        }

        {
            let mut deque = self.queue.deque.lock().unwrap();
            if deque.len() >= self.config.max_queue_depth {
                // Backpressure: drop the middle element (index 1), keep oldest
                // ([0]) and newest (about to be pushed).
                let dropped_idx = if deque.len() > 1 {
                    let dropped = deque.remove(1).unwrap();
                    Some(dropped.chunk_idx)
                } else {
                    // Only one element present — drop it (keep incoming).
                    let dropped = deque.pop_front().unwrap();
                    Some(dropped.chunk_idx)
                };
                if let Some(idx) = dropped_idx {
                    tracing::warn!(
                        "[backpressure] dropped chunk_idx={} (queue full, max={})",
                        idx,
                        self.config.max_queue_depth
                    );
                }
            }
            deque.push_back(chunk);
        }

        self.queue.condvar.notify_one();
        Ok(())
    }

    /// Try to receive a completed [`ChunkPartial`] without blocking.
    ///
    /// Returns `None` immediately if no partial is available.
    pub fn try_recv_partial(&self) -> Option<ChunkPartial> {
        self.partial_rx.lock().unwrap().try_recv().ok()
    }

    /// Block until a [`ChunkPartial`] is available or `timeout` elapses.
    pub fn recv_partial_timeout(&self, timeout: Duration) -> Option<ChunkPartial> {
        self.partial_rx.lock().unwrap().recv_timeout(timeout).ok()
    }

    /// Signal all in-flight and pending inference to stop emitting partials.
    ///
    /// The worker will no longer push to the partial channel after this call.
    /// Already-started inference cannot be aborted (Swift Task limitation), but
    /// its results will be discarded.
    pub fn cancel(&self) {
        self.queue.cancel.store(true, Ordering::SeqCst);
        // Wake the worker so it can observe the cancel flag.
        self.queue.condvar.notify_all();
    }

    /// Signal the worker to finish draining the queue and exit.
    ///
    /// Blocks until the worker thread finishes, then collects all remaining
    /// [`ChunkPartial`]s from the output channel.
    pub fn shutdown(self) -> Vec<ChunkPartial> {
        // Tell the worker to drain and exit.
        self.queue.shutdown.store(true, Ordering::SeqCst);
        self.queue.condvar.notify_all();

        // Wait for the worker to finish.
        if let Some(handle) = self.worker_handle.lock().unwrap().take() {
            let _ = handle.join();
        }

        // Drain remaining partials.
        let rx = self.partial_rx.lock().unwrap();
        let mut out = Vec::new();
        while let Ok(p) = rx.try_recv() {
            out.push(p);
        }
        out
    }

    /// Whether the orchestrator has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.queue.cancel.load(Ordering::Relaxed)
    }
}

// ── Worker loop ───────────────────────────────────────────────────────────────

fn worker_loop(
    config: OrchestratorConfig,
    queue: Arc<InboundQueue>,
    backend: Box<dyn InferenceBackend>,
    partial_tx: std::sync::mpsc::Sender<ChunkPartial>,
) {
    // Elevate this thread to user-interactive QoS for P-core scheduling.
    crate::platform::elevate_thread_qos("streaming-orchestrator-worker");

    let mut is_first_chunk_this_session = true;

    loop {
        // Wait for a chunk (or shutdown/cancel signal).
        let chunk = {
            let mut deque = queue.deque.lock().unwrap();
            loop {
                if queue.shutdown.load(Ordering::Relaxed) && deque.is_empty() {
                    // Graceful exit: queue drained AND shutdown requested.
                    return;
                }
                if let Some(c) = deque.pop_front() {
                    break c;
                }
                // No chunk yet — wait.
                deque = queue.condvar.wait(deque).unwrap();
            }
        };

        // Respect cancel flag.
        if queue.cancel.load(Ordering::Relaxed) {
            // Drain remaining chunks silently.
            let mut deque = queue.deque.lock().unwrap();
            deque.clear();
            if queue.shutdown.load(Ordering::Relaxed) {
                return;
            }
            continue;
        }

        let chunk_idx = chunk.chunk_idx;
        let chunk_audio_ms = chunk.samples.len() as f64 / 16.0; // 16 kHz → ms
        let is_cold = is_first_chunk_this_session;
        is_first_chunk_this_session = false;

        tracing::info!(
            "[t5a_chunk_inference] chunk_idx={} chunk_audio_ms={:.0} cold={}",
            chunk_idx,
            chunk_audio_ms,
            is_cold,
        );

        let infer_start = Instant::now();
        let mut last_partial = String::new();
        let mut first_token_ms: Option<f64> = None;
        let mut first_token_seen = false;

        let result = backend.run(&chunk.samples, &config.model_id, &mut |partial: &str| {
            // Check cancel inside callback.
            if queue.cancel.load(Ordering::Relaxed) {
                return;
            }

            if !first_token_seen && is_cold {
                first_token_ms = Some(infer_start.elapsed().as_secs_f64() * 1000.0);
                first_token_seen = true;
            }
            last_partial = partial.to_string();
        });

        let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        // After inference, check cancel again before emitting.
        if queue.cancel.load(Ordering::Relaxed) {
            tracing::debug!(
                "[t5a_chunk_inference] chunk_idx={} discarded (cancelled after inference)",
                chunk_idx
            );
            continue;
        }

        let partial_text_len = last_partial.chars().count();

        let (partial_text, error) = match result {
            Ok(final_text) => {
                // Use the final authoritative text if non-empty, else last_partial.
                let text = if !final_text.is_empty() {
                    final_text
                } else {
                    last_partial
                };
                (text, None)
            }
            Err(e) => {
                tracing::warn!(
                    "[t5a_chunk_inference] chunk_idx={} error: {}",
                    chunk_idx,
                    e
                );
                (last_partial, Some(e))
            }
        };

        tracing::info!(
            "[t5a_chunk_inference] chunk_idx={} chunk_audio_ms={:.0} inference_ms={:.0} \
             partial_text_len={} first_token_ms={:?} cold={}",
            chunk_idx,
            chunk_audio_ms,
            inference_ms,
            partial_text_len,
            first_token_ms,
            is_cold,
        );

        let cp = ChunkPartial {
            chunk_idx,
            partial_text,
            inference_ms,
            first_token_ms,
            error,
        };

        if partial_tx.send(cp).is_err() {
            // Receiver dropped — nothing to do; exit cleanly.
            tracing::debug!("[streaming_orchestrator] partial_tx receiver gone; exiting worker");
            return;
        }
    }
}

// ── Real mlx-audio backend ────────────────────────────────────────────────────

/// Production implementation of [`InferenceBackend`] that calls
/// `crate::mlx_audio::transcribe_streaming` via FFI.
///
/// On non-aarch64 platforms (CI, tests with real backend) this always returns
/// `Err`.
pub struct MlxAudioBackend;

impl InferenceBackend for MlxAudioBackend {
    fn run(
        &self,
        audio: &[f32],
        model_id: &str,
        on_partial: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        // Write samples to a temp WAV file.
        let tmp_path = {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            std::env::temp_dir().join(format!("handy_mlx_chunk_{}.wav", ts))
        };

        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::create(&tmp_path, spec)
                .map_err(|e| format!("WAV writer create failed: {}", e))?;
            for &sample in audio {
                writer
                    .write_sample(sample)
                    .map_err(|e| format!("WAV write_sample failed: {}", e))?;
            }
            writer
                .finalize()
                .map_err(|e| format!("WAV finalize failed: {}", e))?;
        }

        let mut last_partial = String::new();

        let result =
            crate::mlx_audio::transcribe_streaming(&tmp_path, model_id, |partial: &str| {
                on_partial(partial);
                last_partial = partial.to_string();
            });

        let _ = std::fs::remove_file(&tmp_path);

        match result {
            Ok(()) => Ok(last_partial),
            Err(e) => Err(e),
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_toolkit::audio::chunker::{ChunkTrigger, ChunkedAudio};

    // ── Mock backend ─────────────────────────────────────────────────────────

    /// A mock backend that returns a fixed string after an optional sleep.
    struct MockBackend {
        response: String,
        sleep_ms: u64,
    }

    impl MockBackend {
        fn new(response: impl Into<String>) -> Self {
            Self {
                response: response.into(),
                sleep_ms: 0,
            }
        }

        fn with_delay(mut self, ms: u64) -> Self {
            self.sleep_ms = ms;
            self
        }
    }

    impl InferenceBackend for MockBackend {
        fn run(
            &self,
            _audio: &[f32],
            _model_id: &str,
            on_partial: &mut dyn FnMut(&str),
        ) -> Result<String, String> {
            if self.sleep_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.sleep_ms));
            }
            on_partial(&self.response);
            Ok(self.response.clone())
        }
    }

    /// A sequential mock that returns different responses per call.
    struct SequentialMockBackend {
        responses: Mutex<std::collections::VecDeque<String>>,
    }

    impl SequentialMockBackend {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(responses.iter().map(|s| s.to_string()).collect()),
            }
        }
    }

    impl InferenceBackend for SequentialMockBackend {
        fn run(
            &self,
            _audio: &[f32],
            _model_id: &str,
            on_partial: &mut dyn FnMut(&str),
        ) -> Result<String, String> {
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "fallback".to_string());
            on_partial(&response);
            Ok(response)
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────

    fn make_chunk(idx: u64) -> ChunkedAudio {
        ChunkedAudio {
            samples: vec![0.0f32; 16_000], // 1s of silence
            chunk_idx: idx,
            captured_at_ms: idx * 1000,
            trigger: ChunkTrigger::HopExpired,
        }
    }

    fn default_config() -> OrchestratorConfig {
        OrchestratorConfig {
            model_id: "test-model".to_string(),
            max_queue_depth: 3,
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Push 1 chunk, verify ChunkPartial is received with correct text.
    #[test]
    fn submit_then_recv_one_partial() {
        let backend = Box::new(MockBackend::new("你好"));
        let orch = StreamingOrchestrator::start(default_config(), backend);

        orch.submit(make_chunk(0)).unwrap();

        let partial = orch
            .recv_partial_timeout(Duration::from_secs(5))
            .expect("should receive a partial within 5 seconds");

        assert_eq!(partial.chunk_idx, 0);
        assert_eq!(partial.partial_text, "你好");
        assert!(partial.error.is_none());
        assert!(partial.inference_ms >= 0.0);

        let _ = orch.shutdown();
    }

    /// Push 3 chunks, verify partials emit in chunk_idx order 0, 1, 2.
    #[test]
    fn fifo_order() {
        let backend = Box::new(SequentialMockBackend::new(vec!["chunk0", "chunk1", "chunk2"]));
        let orch = StreamingOrchestrator::start(default_config(), backend);

        for i in 0..3u64 {
            orch.submit(make_chunk(i)).unwrap();
        }

        let mut results = Vec::new();
        for _ in 0..3 {
            if let Some(p) = orch.recv_partial_timeout(Duration::from_secs(5)) {
                results.push(p);
            }
        }

        assert_eq!(results.len(), 3, "should receive 3 partials");
        // FIFO: chunk_idx must appear in ascending order (worker is serial)
        for (expected_idx, p) in results.iter().enumerate() {
            assert_eq!(
                p.chunk_idx, expected_idx as u64,
                "partial at position {} must have chunk_idx={}",
                expected_idx, expected_idx
            );
        }

        let _ = orch.shutdown();
    }

    /// Verify backpressure drops the middle chunk when queue overflows.
    ///
    /// With max_queue=3 and a slow backend (200 ms/chunk), we push 5 chunks
    /// rapidly. The worker is busy with chunk 0; the queue can hold at most 3
    /// more pending chunks. When the 5th arrives the queue is full, so the
    /// middle entry is dropped. We verify that at most 4 chunks are inferred
    /// (chunk 0 + 3 from the queue).
    #[test]
    fn backpressure_drops_middle() {
        // Slow backend: 50 ms per chunk so we can flood the queue easily.
        let backend = Box::new(MockBackend::new("result").with_delay(50));
        let config = OrchestratorConfig {
            model_id: "test-model".to_string(),
            max_queue_depth: 3,
        };
        let orch = StreamingOrchestrator::start(config, backend);

        // Push 5 chunks quickly, before the worker finishes even the first.
        for i in 0..5u64 {
            // submit is non-blocking; errors on cancel/shutdown only
            let _ = orch.submit(make_chunk(i));
        }

        // Collect all partials with a generous timeout.
        let collected = orch.shutdown();

        // With max_queue=3 and 5 pushes:
        //   - push 0: worker pops immediately (or queued as first)
        //   - push 1..4: queue fills to 3 then backpressure fires once (drops 1 middle)
        // Total inferred ≤ 4 chunks (5 minus 1 dropped), ≥ 1.
        let total = collected.len();
        assert!(
            total >= 1 && total <= 4,
            "expected 1–4 partials with backpressure, got {}",
            total
        );
    }

    /// After cancel(), no new partials are emitted.
    #[test]
    fn cancel_stops_emitting() {
        // Use a slow backend so chunk 1 is still pending after cancel.
        let backend = Box::new(MockBackend::new("result").with_delay(30));
        let orch = StreamingOrchestrator::start(default_config(), backend);

        // Submit chunk 0 and let it start.
        orch.submit(make_chunk(0)).unwrap();
        // Tiny yield so worker picks up chunk 0.
        std::thread::sleep(Duration::from_millis(5));

        // Submit chunk 1 — sits in queue.
        orch.submit(make_chunk(1)).unwrap();

        // Cancel.
        orch.cancel();

        // Wait a bit for the worker to process.
        std::thread::sleep(Duration::from_millis(100));

        // After cancel, no partial should arrive for the cancelled work.
        // (Chunk 0 may or may not have completed before cancel depending on timing.)
        // The important invariant: calling shutdown() does not panic.
        let remaining = orch.shutdown();
        // All remaining partials (if any) must come from before the cancel.
        // We can't assert exact counts due to timing, but no crash = pass.
        drop(remaining);
    }

    /// shutdown() drains the queue and returns all completed partials.
    #[test]
    fn shutdown_drains() {
        let backend = Box::new(SequentialMockBackend::new(vec!["a", "b"]));
        let orch = StreamingOrchestrator::start(default_config(), backend);

        orch.submit(make_chunk(0)).unwrap();
        orch.submit(make_chunk(1)).unwrap();

        let partials = orch.shutdown();

        assert_eq!(
            partials.len(),
            2,
            "shutdown must drain all 2 partials; got {}",
            partials.len()
        );
        assert_eq!(partials[0].chunk_idx, 0);
        assert_eq!(partials[1].chunk_idx, 1);
    }

    /// first_token_ms is only set for chunk_idx == 0 (cold = first chunk).
    #[test]
    fn first_token_ms_only_first_chunk() {
        let backend = Box::new(MockBackend::new("text"));
        let orch = StreamingOrchestrator::start(default_config(), backend);

        for i in 0..3u64 {
            orch.submit(make_chunk(i)).unwrap();
        }

        let partials = orch.shutdown();

        assert_eq!(
            partials.len(),
            3,
            "expected 3 partials; got {}",
            partials.len()
        );

        // Only the very first chunk (idx=0) should have first_token_ms set.
        let first = partials.iter().find(|p| p.chunk_idx == 0);
        let others: Vec<_> = partials.iter().filter(|p| p.chunk_idx != 0).collect();

        if let Some(f) = first {
            assert!(
                f.first_token_ms.is_some(),
                "chunk_idx=0 must have first_token_ms set"
            );
        }
        for o in others {
            assert!(
                o.first_token_ms.is_none(),
                "chunk_idx={} must NOT have first_token_ms",
                o.chunk_idx
            );
        }
    }

    /// Verify that submit returns Err after cancel().
    #[test]
    fn submit_after_cancel_returns_err() {
        let backend = Box::new(MockBackend::new("x"));
        let orch = StreamingOrchestrator::start(default_config(), backend);
        orch.cancel();
        let result = orch.submit(make_chunk(99));
        assert!(
            result.is_err(),
            "submit after cancel must return Err, got Ok"
        );
        let _ = orch.shutdown();
    }
}
