//! FD-006 M0 — VAD-aware AudioChunker
//!
//! Maintains a ringbuffer of recent 16 kHz f32 audio and emits sliding-window
//! chunks to downstream inference.  Emission is triggered by whichever comes
//! first:
//!
//!  1. VAD-detected silence ≥ `vad_silence_ms`  (word-boundary friendly)
//!  2. Fallback: buffer already holds ≥ `window_ms` audio AND time since last
//!     emit ≥ `hop_ms`  (keeps cadence bounded)
//!
//! Consecutive chunks share `overlap_ms` of audio to avoid word-boundary cuts.

use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the chunker.  All durations refer to audio time at the
/// `sample_rate` specified here.
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Length of each emitted chunk (main body, **excluding** overlap prefix).
    pub window_ms: u64,
    /// Minimum time between two consecutive emits triggered by hop (not VAD).
    pub hop_ms: u64,
    /// How many ms of the previous chunk's tail to prepend to the next chunk,
    /// preventing word-boundary cuts.
    pub overlap_ms: u64,
    /// Consecutive silence that triggers an early emit (VAD path).
    pub vad_silence_ms: u64,
    /// Must be 16 000 — the native ASR sample rate.
    pub sample_rate: u32,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            window_ms: 2000,
            hop_ms: 1500,
            overlap_ms: 300,
            vad_silence_ms: 200,
            sample_rate: 16_000,
        }
    }
}

/// What caused this chunk to be emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkTrigger {
    /// VAD detected ≥ `vad_silence_ms` of consecutive silence.
    VadSilence,
    /// Fallback: `hop_ms` elapsed and buffer was full enough.
    HopExpired,
    /// Caller called `flush_on_session_end()`.
    SessionEnd,
}

/// A single emitted audio chunk.
#[derive(Debug, Clone)]
pub struct ChunkedAudio {
    /// PCM samples at `config.sample_rate`.  Includes `overlap_ms` prefix from
    /// the previous chunk.
    pub samples: Vec<f32>,
    /// Zero-based monotonic index within the current session.
    pub chunk_idx: u64,
    /// Time of the **first** sample in this chunk, relative to session start
    /// (milliseconds).
    pub captured_at_ms: u64,
    /// What triggered this emission.
    pub trigger: ChunkTrigger,
}

// ---------------------------------------------------------------------------
// AudioChunker
// ---------------------------------------------------------------------------

pub struct AudioChunker {
    config: ChunkerConfig,

    /// Ringbuffer — stores all samples since the last emit (minus the overlap
    /// that was drained).  Index 0 = oldest.
    ringbuffer: VecDeque<f32>,

    /// Wall-clock instant when this session started (first `ingest` call).
    session_start: Option<Instant>,

    /// Wall-clock instant of the last emit (or session start).
    last_emit_at: Instant,

    /// How many milliseconds of consecutive silence we have seen since the
    /// last speech frame.
    vad_silence_run_ms: u64,

    /// FD-006 follow-up #5: did we ingest at least one speech frame since the
    /// last emit?  Used to suppress repeated silence-only chunks (which the
    /// model hallucinates as "嗯。" filler tokens) when the user finishes
    /// speaking but keeps holding the push-to-talk key.
    had_speech_since_last_emit: bool,

    /// Total number of chunks emitted in this session.
    chunk_idx: u64,

    /// Total audio ingested (in samples) since session start — used to
    /// compute `captured_at_ms` for each chunk.
    total_samples_ingested: u64,

    /// Sample position (from session start) at which the current ringbuffer
    /// head starts.  We advance this whenever we drain the front of the
    /// ringbuffer.
    ringbuffer_start_sample: u64,
}

impl AudioChunker {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(config: ChunkerConfig) -> Self {
        let now = Instant::now();
        Self {
            config,
            ringbuffer: VecDeque::new(),
            session_start: None,
            last_emit_at: now,
            vad_silence_run_ms: 0,
            had_speech_since_last_emit: false,
            chunk_idx: 0,
            total_samples_ingested: 0,
            ringbuffer_start_sample: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    #[inline]
    fn ms_to_samples(&self, ms: u64) -> usize {
        (self.config.sample_rate as u64 * ms / 1000) as usize
    }

    #[inline]
    fn samples_to_ms(&self, n: usize) -> u64 {
        n as u64 * 1000 / self.config.sample_rate as u64
    }

    /// Build a `ChunkedAudio` from the tail of the ringbuffer and advance the
    /// drain cursor (keeping `overlap_ms` worth of samples for the next chunk).
    fn emit_from_ringbuffer(&mut self, trigger: ChunkTrigger) -> ChunkedAudio {
        let window_samples = self.ms_to_samples(self.config.window_ms);
        let overlap_samples = self.ms_to_samples(self.config.overlap_ms);

        // Take at most (window + overlap) samples from the end of the buffer.
        // If the buffer is shorter than that, take everything.
        let total_in_buf = self.ringbuffer.len();
        let take = (window_samples + overlap_samples).min(total_in_buf);

        // Start index within ringbuffer from which we copy.
        let copy_start = total_in_buf.saturating_sub(take);

        // `captured_at_ms` = absolute sample index of the first copied sample,
        // divided back to ms.
        let first_sample_abs = self.ringbuffer_start_sample + copy_start as u64;
        let captured_at_ms = first_sample_abs * 1000 / self.config.sample_rate as u64;

        // Copy the window into a Vec.
        let samples: Vec<f32> = self.ringbuffer.range(copy_start..).copied().collect();

        // Drain the front up to (but not including) the last `overlap_samples`
        // worth — those are retained for the next chunk's prefix.
        let drain_count = total_in_buf.saturating_sub(overlap_samples);
        let actually_drained: Vec<_> = self.ringbuffer.drain(..drain_count).collect();
        let _ = actually_drained; // drop
        self.ringbuffer_start_sample += drain_count as u64;

        let idx = self.chunk_idx;
        self.chunk_idx += 1;
        self.last_emit_at = Instant::now();
        self.vad_silence_run_ms = 0;
        self.had_speech_since_last_emit = false;

        ChunkedAudio {
            samples,
            chunk_idx: idx,
            captured_at_ms,
            trigger,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Feed a new audio frame from the recording pipeline.
    ///
    /// `is_speech` should be the VAD decision for **this** frame.  Callers
    /// typically derive it from `VadFrame::is_speech()` but can also pass
    /// `true` to disable VAD-triggered early emission.
    ///
    /// Returns `Some(ChunkedAudio)` when an emit condition is met; `None`
    /// otherwise.
    pub fn ingest(&mut self, samples: &[f32], is_speech: bool) -> Option<ChunkedAudio> {
        // Initialise session clock on first ingest.
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
            self.last_emit_at = Instant::now();
        }

        // Append new samples to ringbuffer.
        self.ringbuffer.extend(samples.iter().copied());
        self.total_samples_ingested += samples.len() as u64;

        // Update VAD silence tracker.
        let frame_ms = self.samples_to_ms(samples.len());
        if is_speech {
            self.vad_silence_run_ms = 0;
            self.had_speech_since_last_emit = true;
        } else {
            self.vad_silence_run_ms += frame_ms;
        }

        // ---- Emit decision -------------------------------------------------

        let buf_ms = self.samples_to_ms(self.ringbuffer.len());
        let half_window_ms = self.config.window_ms / 2;

        // FD-006 follow-up #5: suppress chunks where no speech occurred since
        // the last emit. Without this, a user holding push-to-talk silently
        // keeps emitting silent chunks every hop_ms, which the ASR model
        // hallucinates as "嗯。" filler — visible to the user as endless
        // "嗯。嗯。嗯。" being typed after they stopped speaking.
        if !self.had_speech_since_last_emit {
            return None;
        }

        // VAD path: silence ≥ threshold AND we have meaningful audio.
        if self.vad_silence_run_ms >= self.config.vad_silence_ms && buf_ms >= half_window_ms {
            return Some(self.emit_from_ringbuffer(ChunkTrigger::VadSilence));
        }

        // Hop path: buffer is full AND hop interval expired.
        let elapsed_ms = self.last_emit_at.elapsed().as_millis() as u64;
        if elapsed_ms >= self.config.hop_ms && buf_ms >= self.config.window_ms {
            return Some(self.emit_from_ringbuffer(ChunkTrigger::HopExpired));
        }

        None
    }

    /// Called when recording stops.  Emits all residual audio that contains
    /// meaningful speech (≥ 200 ms of non-silence).
    ///
    /// May return multiple chunks if the residual buffer exceeds `window_ms`.
    pub fn flush_on_session_end(&mut self) -> Vec<ChunkedAudio> {
        const MIN_RESIDUAL_MS: u64 = 200;

        let mut out = Vec::new();

        loop {
            let buf_len = self.ringbuffer.len();
            let buf_ms = self.samples_to_ms(buf_len);

            // Stop when nothing meaningful is left.
            if buf_ms < MIN_RESIDUAL_MS {
                self.ringbuffer.clear();
                break;
            }

            // If the entire residual is silence, discard it.
            // We approximate: if vad_silence_run_ms covers the whole buffer,
            // the buffer is effectively silent.
            if self.vad_silence_run_ms >= buf_ms {
                self.ringbuffer.clear();
                break;
            }

            // FD-006 follow-up #5: if no speech was ingested since the last
            // emit, the residual is silence-only padding (model would
            // hallucinate "嗯。"). Skip.
            if !self.had_speech_since_last_emit {
                self.ringbuffer.clear();
                break;
            }

            let window_samples = self.ms_to_samples(self.config.window_ms);
            let overlap_samples = self.ms_to_samples(self.config.overlap_ms);

            if buf_len <= overlap_samples {
                // Only overlap-sized tail remains — not worth a new chunk, discard.
                self.ringbuffer.clear();
                break;
            }

            // Emit the current buffer chunk.
            let chunk = self.emit_from_ringbuffer(ChunkTrigger::SessionEnd);
            out.push(chunk);

            // After emit, `emit_from_ringbuffer` retains `overlap_samples` in
            // the ringbuffer and resets `vad_silence_run_ms = 0`.  If the
            // remaining buffer is only the overlap tail, we must not loop again
            // (the overlap itself is < MIN_RESIDUAL_MS threshold when
            // overlap_ms < 200 ms, but if overlap_ms >= 200 ms it would loop).
            // Safety net: if what's left is ≤ overlap_samples, stop.
            let remaining_ms = self.samples_to_ms(self.ringbuffer.len());
            if remaining_ms < MIN_RESIDUAL_MS
                || self.ringbuffer.len() <= overlap_samples
                || (buf_len <= window_samples + overlap_samples)
            {
                // We just emitted the only (or last) meaningful chunk.
                self.ringbuffer.clear();
                break;
            }
        }

        out
    }

    /// Current chunk index (= number of chunks emitted so far).
    pub fn current_chunk_idx(&self) -> u64 {
        self.chunk_idx
    }

    /// Reset all state — next recording session starts fresh.
    pub fn reset(&mut self) {
        self.ringbuffer.clear();
        self.session_start = None;
        self.last_emit_at = Instant::now();
        self.vad_silence_run_ms = 0;
        self.had_speech_since_last_emit = false;
        self.chunk_idx = 0;
        self.total_samples_ingested = 0;
        self.ringbuffer_start_sample = 0;
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal config with a fast hop so tests are sub-ms.
    fn cfg() -> ChunkerConfig {
        ChunkerConfig {
            window_ms: 2000,
            hop_ms: 1500,
            overlap_ms: 300,
            vad_silence_ms: 200,
            sample_rate: 16_000,
        }
    }

    /// Generate `ms` milliseconds of mono f32 audio at 16 kHz.
    fn silence(ms: u64) -> Vec<f32> {
        let n = (16_000u64 * ms / 1000) as usize;
        vec![0.0f32; n]
    }

    fn speech(ms: u64) -> Vec<f32> {
        let n = (16_000u64 * ms / 1000) as usize;
        // Non-zero so callers can distinguish from silence.
        vec![0.5f32; n]
    }

    // -----------------------------------------------------------------------

    #[test]
    fn no_emit_below_hop_threshold() {
        let mut ch = AudioChunker::new(cfg());
        // Push 500 ms of speech — well below hop_ms (1500 ms) and window_ms (2000 ms).
        let result = ch.ingest(&speech(500), true);
        assert!(result.is_none(), "should not emit before hop threshold");
    }

    #[test]
    fn emit_at_hop_expired_with_full_window() {
        // Use hop_ms = 0 so the check fires immediately without a real sleep.
        let fast_cfg = ChunkerConfig {
            hop_ms: 0,
            ..cfg()
        };
        let mut ch = AudioChunker::new(fast_cfg);

        // Push window_ms (2000 ms) of speech.
        let result = ch.ingest(&speech(2000), true);
        assert!(result.is_some(), "should emit after hop=0 + full window");
        let chunk = result.unwrap();
        assert_eq!(chunk.trigger, ChunkTrigger::HopExpired);
        assert_eq!(chunk.chunk_idx, 0);
    }

    #[test]
    fn vad_silence_triggers_early_emit() {
        let mut ch = AudioChunker::new(cfg());

        // Push 700 ms of speech first (above half_window = 1000 ms? No, 700 < 1000).
        // Use half_window_ms = window_ms/2 = 1000, so we need at least 1000 ms.
        ch.ingest(&speech(1000), true);

        // Now push 250 ms of silence — exceeds vad_silence_ms (200 ms).
        let result = ch.ingest(&silence(250), false);
        assert!(
            result.is_some(),
            "VAD silence should trigger early emit after 250 ms silence"
        );
        let chunk = result.unwrap();
        assert_eq!(chunk.trigger, ChunkTrigger::VadSilence);
    }

    #[test]
    fn vad_silence_too_short_no_early_emit() {
        let mut ch = AudioChunker::new(cfg());

        ch.ingest(&speech(1000), true);

        // Push 100 ms of silence — below vad_silence_ms (200 ms).
        let result = ch.ingest(&silence(100), false);
        assert!(
            result.is_none(),
            "100 ms silence should NOT trigger early emit (below vad_silence_ms)"
        );
    }

    #[test]
    fn overlap_correctness() {
        let overlap_ms = 300u64;
        let overlap_samples = (16_000u64 * overlap_ms / 1000) as usize;

        // Use hop_ms=0 so we can emit deterministically.
        let fast_cfg = ChunkerConfig {
            hop_ms: 0,
            overlap_ms,
            ..cfg()
        };
        let mut ch = AudioChunker::new(fast_cfg);

        // Fill with identifiable values: first window with 1.0, second with 2.0.
        let first_block = vec![1.0f32; 16_000 * 2]; // 2000 ms
        let second_block = vec![2.0f32; 16_000 * 2]; // 2000 ms

        let chunk0 = ch.ingest(&first_block, true).expect("chunk 0 should emit");

        // After chunk0 emits, the ringbuffer retains `overlap_samples` worth of
        // 1.0 samples.  Push second block.
        let chunk1 = ch.ingest(&second_block, true).expect("chunk 1 should emit");

        // The first `overlap_samples` of chunk1 should equal the last
        // `overlap_samples` of chunk0.
        let tail_of_0 = &chunk0.samples[chunk0.samples.len() - overlap_samples..];
        let head_of_1 = &chunk1.samples[..overlap_samples];

        assert_eq!(
            tail_of_0, head_of_1,
            "overlap region must be identical between consecutive chunks"
        );
    }

    #[test]
    fn chunk_idx_monotonic() {
        let fast_cfg = ChunkerConfig {
            hop_ms: 0,
            ..cfg()
        };
        let mut ch = AudioChunker::new(fast_cfg);

        let mut last_idx: Option<u64> = None;
        for _ in 0..3 {
            if let Some(chunk) = ch.ingest(&speech(2000), true) {
                if let Some(prev) = last_idx {
                    assert_eq!(
                        chunk.chunk_idx,
                        prev + 1,
                        "chunk_idx must be strictly monotonic"
                    );
                }
                last_idx = Some(chunk.chunk_idx);
            }
        }
        assert!(last_idx.is_some(), "at least one chunk should have been emitted");
    }

    #[test]
    fn flush_on_session_end_emits_residual() {
        let mut ch = AudioChunker::new(cfg());

        // Push 800 ms of speech (no emit yet — below hop_ms).
        ch.ingest(&speech(800), true);

        let chunks = ch.flush_on_session_end();
        assert_eq!(chunks.len(), 1, "flush should emit exactly 1 residual chunk");
        assert_eq!(chunks[0].trigger, ChunkTrigger::SessionEnd);
    }

    #[test]
    fn flush_drops_silent_residual() {
        let mut ch = AudioChunker::new(cfg());

        // Push only 100 ms of silence — below MIN_RESIDUAL_MS (200 ms).
        ch.ingest(&silence(100), false);

        let chunks = ch.flush_on_session_end();
        assert!(
            chunks.is_empty(),
            "flush should drop residual that is below 200 ms"
        );
    }

    #[test]
    fn flush_drops_all_silence_buffer() {
        let mut ch = AudioChunker::new(cfg());

        // Push 500 ms silence — above MIN_RESIDUAL_MS in size, but entirely silent.
        // vad_silence_run_ms will equal buf_ms, so flush should discard.
        ch.ingest(&silence(500), false);

        let chunks = ch.flush_on_session_end();
        assert!(
            chunks.is_empty(),
            "flush should discard a buffer that is entirely silent"
        );
    }

    #[test]
    fn reset_clears_state() {
        let fast_cfg = ChunkerConfig {
            hop_ms: 0,
            ..cfg()
        };
        let mut ch = AudioChunker::new(fast_cfg);

        // Emit at least one chunk.
        ch.ingest(&speech(2000), true);
        assert!(ch.current_chunk_idx() > 0, "should have emitted before reset");

        ch.reset();

        assert_eq!(ch.current_chunk_idx(), 0, "chunk_idx should be 0 after reset");
        assert!(
            ch.ringbuffer.is_empty(),
            "ringbuffer should be empty after reset"
        );
        assert_eq!(
            ch.vad_silence_run_ms, 0,
            "silence counter should be 0 after reset"
        );
    }

    #[test]
    fn captured_at_ms_correct() {
        // Use hop_ms=0 so we emit immediately.
        let fast_cfg = ChunkerConfig {
            hop_ms: 0,
            ..cfg()
        };
        let mut ch = AudioChunker::new(fast_cfg);

        // First chunk: ingest 2000 ms of speech.
        let chunk0 = ch.ingest(&speech(2000), true).expect("should emit chunk 0");
        // The chunk captures from the very start of the buffer → captured_at_ms = 0.
        assert_eq!(
            chunk0.captured_at_ms, 0,
            "first chunk should start at ms=0 (relative to session start)"
        );
    }

    #[test]
    fn multiple_chunks_after_flush() {
        // Regression: flush should handle buffer > window_ms by emitting multiple chunks.
        let mut ch = AudioChunker::new(cfg());

        // Use hop_ms=0 to force emit on every push that fills the window.
        let fast_cfg = ChunkerConfig {
            hop_ms: 0,
            ..cfg()
        };
        let mut ch = AudioChunker::new(fast_cfg);

        // Push 500 ms increments — each push accumulates audio; once we exceed
        // window_ms (2000 ms) with hop=0 the next push that crosses the threshold
        // triggers an emit.
        for _ in 0..10 {
            ch.ingest(&speech(500), true);
        }
        // After 10 × 500 ms = 5000 ms pushed with hop=0 at least 2 chunks emitted.
        let total_emitted = ch.current_chunk_idx();
        assert!(
            total_emitted >= 2,
            "expected ≥2 emits for 5000 ms with hop=0, got {total_emitted}"
        );

        // Now flush residual — must not infinite-loop.
        let _ = ch.flush_on_session_end();
    }
}
