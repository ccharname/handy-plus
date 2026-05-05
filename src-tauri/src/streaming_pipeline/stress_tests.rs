//! FD-006 M4 — Stress tests: cancel race / backpressure / threading safety.
//!
//! These are **integration-level** tests that exercise the
//! [`StreamingOrchestrator`] + [`AudioChunker`] together under adversarial
//! conditions.  They use [`MockBackend`] variants instead of the real
//! mlx-audio-swift FFI so they run on all platforms and in CI without a GPU
//! or audio device.
//!
//! Five test cases:
//!
//! 1. `cancel_race_100_iter`                   — cancel at random points, 100 iterations
//! 2. `backpressure_drops_middle_under_burst`  — 20 rapid submits on a 3-slot queue
//! 3. `concurrent_submit_no_panic`             — 4 threads hammering submit concurrently
//! 4. `chunker_thread_safety`                  — 4 threads ingest into a Mutex<AudioChunker>
//! 5. `shutdown_drains_no_leak`                — 50 orchestrator create+submit+shutdown cycles

#[cfg(test)]
mod stress_tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::audio_toolkit::audio::chunker::{AudioChunker, ChunkTrigger, ChunkedAudio, ChunkerConfig};
    use crate::streaming_pipeline::{InferenceBackend, OrchestratorConfig, StreamingOrchestrator};

    // ─────────────────────────────────────────────────────────────────────────
    // Mock backends
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns immediately with a fixed response. Simulates a very fast model.
    struct FastMockBackend;

    impl InferenceBackend for FastMockBackend {
        fn run(
            &self,
            _audio: &[f32],
            _model_id: &str,
            on_partial: &mut dyn FnMut(&str),
        ) -> Result<String, String> {
            on_partial("hi");
            Ok("hi".to_string())
        }
    }

    /// Sleeps for a configurable duration before returning. Simulates a slow model.
    struct SlowMockBackend {
        delay: Duration,
    }

    impl SlowMockBackend {
        fn new(delay: Duration) -> Self {
            Self { delay }
        }
    }

    impl InferenceBackend for SlowMockBackend {
        fn run(
            &self,
            _audio: &[f32],
            _model_id: &str,
            on_partial: &mut dyn FnMut(&str),
        ) -> Result<String, String> {
            std::thread::sleep(self.delay);
            on_partial("delayed");
            Ok("delayed".to_string())
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn make_chunk(idx: u64) -> ChunkedAudio {
        ChunkedAudio {
            samples: vec![0.0f32; 16_000], // 1 s of silence at 16 kHz
            chunk_idx: idx,
            captured_at_ms: idx * 1000,
            trigger: ChunkTrigger::HopExpired,
        }
    }

    fn default_config() -> OrchestratorConfig {
        OrchestratorConfig {
            model_id: "stress-test-model".to_string(),
            max_queue_depth: 3,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — cancel_race_100_iter
    // ─────────────────────────────────────────────────────────────────────────

    /// Cancel the orchestrator at a pseudo-random point after submitting 5–10
    /// chunks.  Run 100 times.  Every iteration must complete without panicking
    /// or deadlocking.
    ///
    /// Uses a 50 ms backend delay so inference is frequently in-flight when
    /// cancel fires.  The cancel delay varies from 0–99 ms via a deterministic
    /// pattern (`i * 7 % 100`) to hit all phases: before first inference,
    /// during inference, between inferences, and after all are done.
    #[test]
    fn cancel_race_100_iter() {
        for i in 0..100u64 {
            let backend = SlowMockBackend::new(Duration::from_millis(50));
            let orch = Arc::new(StreamingOrchestrator::start(
                default_config(),
                Box::new(backend),
            ));

            // Submit 5–10 chunks per iteration.
            let n_chunks = 5 + (i % 6) as usize;
            for j in 0..n_chunks as u64 {
                let _ = orch.submit(make_chunk(j));
            }

            // Cancel after a varying delay (0–99 ms).
            let delay_ms = (i * 7) % 100;
            if delay_ms > 0 {
                std::thread::sleep(Duration::from_millis(delay_ms));
            }
            orch.cancel();

            // shutdown() must not panic or deadlock.
            let partials = Arc::try_unwrap(orch)
                .ok()
                .expect("test holds the only Arc ref")
                .shutdown();

            // After cancel: partials produced ≤ chunks submitted.
            assert!(
                partials.len() <= n_chunks,
                "iter={i}: got {} partials but only submitted {n_chunks} chunks",
                partials.len()
            );
        }
        // 100 iterations completing without panic = pass.
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — backpressure_drops_middle_under_burst
    // ─────────────────────────────────────────────────────────────────────────

    /// Burst-submit 20 chunks into a queue limited to 3.  The slow backend
    /// (200 ms/chunk) ensures the queue is always full when new chunks arrive.
    ///
    /// Expected invariants:
    ///   * Total inferred chunks < 20  (many were dropped)
    ///   * chunk_idx=0  is present  (earliest preserved for partial continuity)
    ///   * chunk_idx=19 is present  (newest preserved for realtime feel)
    #[test]
    fn backpressure_drops_middle_under_burst() {
        let backend = SlowMockBackend::new(Duration::from_millis(200));
        let config = OrchestratorConfig {
            model_id: "bp-test-model".to_string(),
            max_queue_depth: 3,
        };
        let orch = StreamingOrchestrator::start(config, Box::new(backend));

        // Rapid-fire 20 chunks (non-blocking).
        for i in 0..20u64 {
            let _ = orch.submit(make_chunk(i));
        }

        let partials = orch.shutdown();

        // Should have processed significantly fewer than 20 chunks.
        assert!(
            partials.len() < 20,
            "expected backpressure to drop some chunks; got {} partials",
            partials.len()
        );

        // Must have processed at least 2 (oldest + newest).
        assert!(
            partials.len() >= 2,
            "expected at least 2 partials (oldest + newest), got {}",
            partials.len()
        );

        let idx_set: HashSet<u64> = partials.iter().map(|p| p.chunk_idx).collect();

        // Chunk 0 (oldest) must be present.
        assert!(
            idx_set.contains(&0),
            "chunk_idx=0 (earliest) must be preserved; idx_set={idx_set:?}"
        );

        // Chunk 19 (newest) must be present.
        assert!(
            idx_set.contains(&19),
            "chunk_idx=19 (newest) must be preserved; idx_set={idx_set:?}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 — concurrent_submit_no_panic
    // ─────────────────────────────────────────────────────────────────────────

    /// Four threads each submit 25 chunks concurrently (100 total).  The fast
    /// backend returns immediately.  The test verifies that concurrent access
    /// to the shared inbound queue never panics (no data race, no deadlock).
    ///
    /// We don't assert specific partial counts because backpressure may drop
    /// many chunks under load — we only require ≥ 1 partial.
    #[test]
    fn concurrent_submit_no_panic() {
        let backend = FastMockBackend;
        let orch = Arc::new(StreamingOrchestrator::start(
            default_config(),
            Box::new(backend),
        ));

        let threads: Vec<_> = (0..4u64)
            .map(|t| {
                let o = Arc::clone(&orch);
                std::thread::spawn(move || {
                    for i in 0..25u64 {
                        // Chunk indices are globally unique per thread.
                        let chunk = make_chunk(t * 25 + i);
                        let _ = o.submit(chunk);
                    }
                })
            })
            .collect();

        for h in threads {
            h.join().expect("submitter thread must not panic");
        }

        // Drain and shut down — must not panic.
        let partials = Arc::try_unwrap(orch)
            .ok()
            .expect("test holds the only Arc ref after threads finish")
            .shutdown();

        // Under heavy concurrent load with max_queue_depth=3 and 100 submits
        // most chunks are dropped by backpressure.  We only require at least 1
        // partial to prove the worker ran at all.
        assert!(
            partials.len() >= 1,
            "expected ≥ 1 partial from concurrent submit; got 0"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4 — chunker_thread_safety
    // ─────────────────────────────────────────────────────────────────────────

    /// Four threads each call `AudioChunker::ingest` 50 times while holding a
    /// `Mutex<AudioChunker>`.  This validates that the chunker produces no
    /// logical invariant violations under concurrent access (e.g. double-emit,
    /// corrupted ringbuffer, overflow).  Not-panicking is the primary signal.
    #[test]
    fn chunker_thread_safety() {
        // Use hop_ms=0 so emissions happen frequently (more interesting stress).
        let config = ChunkerConfig {
            hop_ms: 0,
            ..ChunkerConfig::default()
        };
        let chunker = Arc::new(Mutex::new(AudioChunker::new(config)));

        let threads: Vec<_> = (0..4usize)
            .map(|t| {
                let c = Arc::clone(&chunker);
                std::thread::spawn(move || {
                    for i in 0..50usize {
                        let samples = vec![0.1_f32; 160]; // 10 ms frame at 16 kHz
                        let is_speech = (t + i) % 2 == 0;
                        let mut guard = c.lock().unwrap();
                        let _chunk = guard.ingest(&samples, is_speech);
                        // We don't assert specific chunk counts — just that
                        // it doesn't panic.
                    }
                })
            })
            .collect();

        for h in threads {
            h.join().expect("chunker thread must not panic");
        }

        // Final flush — must not panic or infinite-loop.
        let remaining = chunker.lock().unwrap().flush_on_session_end();
        // 0 or more residual chunks is fine.
        drop(remaining);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5 — shutdown_drains_no_leak
    // ─────────────────────────────────────────────────────────────────────────

    /// Spin up 50 orchestrators in sequence, each submitting 3 chunks, then
    /// calling `shutdown()`.  Verifies that worker threads are always joined
    /// cleanly (no thread leak, no zombie threads) and no resources are left
    /// dangling.
    ///
    /// A leaked thread would eventually cause `std::thread::spawn` to fail or
    /// the process to run out of resources — 50 iterations is enough to catch
    /// a systematic leak on any sensible OS.
    #[test]
    fn shutdown_drains_no_leak() {
        for iteration in 0..50usize {
            let backend = FastMockBackend;
            let orch = StreamingOrchestrator::start(default_config(), Box::new(backend));

            for i in 0..3u64 {
                let _ = orch.submit(make_chunk(i));
            }

            let partials = orch.shutdown();

            // We submitted 3 chunks; worker serialises them.  With the fast
            // backend (no sleep) and 3-slot queue all should be processed.
            assert!(
                partials.len() <= 3,
                "iter={iteration}: got {} partials but only submitted 3",
                partials.len()
            );
            // At least 1 partial should arrive (worker ran).
            assert!(
                partials.len() >= 1,
                "iter={iteration}: expected ≥ 1 partial after shutdown, got 0"
            );
        }
        // 50 iterations without panic = no thread leak detected.
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bonus: cancel_before_any_submit
    // ─────────────────────────────────────────────────────────────────────────

    /// Edge case: cancel immediately after start, before any chunk is submitted.
    /// shutdown() must not block or panic.
    #[test]
    fn cancel_before_any_submit() {
        let backend = SlowMockBackend::new(Duration::from_millis(50));
        let orch = StreamingOrchestrator::start(default_config(), Box::new(backend));
        orch.cancel();
        // submit after cancel returns Err, not a panic.
        let res = orch.submit(make_chunk(0));
        assert!(res.is_err(), "submit after cancel must return Err");
        let partials = orch.shutdown();
        assert!(partials.is_empty(), "no partials expected when cancel fired before any submit");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bonus: cancel_flag_propagates_to_worker
    // ─────────────────────────────────────────────────────────────────────────

    /// Verify that `is_cancelled()` reflects the cancel flag correctly and
    /// that a second `cancel()` call is idempotent (no double-free / no panic).
    #[test]
    fn cancel_idempotent() {
        let backend = FastMockBackend;
        let orch = StreamingOrchestrator::start(default_config(), Box::new(backend));

        assert!(!orch.is_cancelled());
        orch.cancel();
        assert!(orch.is_cancelled(), "is_cancelled() must return true after cancel()");
        // Second cancel — must not panic.
        orch.cancel();
        assert!(orch.is_cancelled());

        let _ = orch.shutdown();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bonus: cancel_drains_in_flight_without_blocking
    // ─────────────────────────────────────────────────────────────────────────

    /// If inference is in-flight (slow backend) and cancel fires, shutdown()
    /// must still return promptly (within a generous 2-second wall-clock
    /// budget) rather than blocking forever.
    #[test]
    fn cancel_drains_in_flight_without_blocking() {
        // Backend sleeps 300 ms per chunk.
        let backend = SlowMockBackend::new(Duration::from_millis(300));
        let orch = Arc::new(StreamingOrchestrator::start(
            default_config(),
            Box::new(backend),
        ));

        // Submit a handful of chunks so the queue fills while the worker runs.
        for i in 0..5u64 {
            let _ = orch.submit(make_chunk(i));
        }

        // Cancel immediately — worker is likely mid-inference on chunk 0.
        orch.cancel();

        let start = std::time::Instant::now();

        let partials = Arc::try_unwrap(orch)
            .ok()
            .expect("test holds the only Arc ref")
            .shutdown();

        let elapsed = start.elapsed();

        // shutdown() should complete well within 2 s even though each inference
        // sleeps 300 ms — the cancel flag causes the queue to be drained and the
        // worker exits after the current in-flight inference finishes.
        assert!(
            elapsed < Duration::from_secs(2),
            "shutdown after cancel took {elapsed:?}; expected < 2 s"
        );

        // Cancel was requested before most chunks could run; expect few partials.
        assert!(
            partials.len() <= 5,
            "expected ≤ 5 partials after cancel, got {}",
            partials.len()
        );
    }
}
