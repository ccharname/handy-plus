//! Silence/noise pre-filter shared by ASR engines that hallucinate non-empty
//! text on inputs without speech (SenseVoice → "我。"/"嗯。", FunASR-Nano →
//! token streams from the LLM decoder).
//!
//! A single Silero pass over 30 ms frames is several orders of magnitude
//! cheaper than running the full ASR engine on a non-speech clip, and lets
//! the engine skip work entirely when fewer than `MIN_VOICE_FRAMES` frames
//! are voiced.

use crate::audio_toolkit::vad::VoiceActivityDetector;
use crate::audio_toolkit::SileroVad;
use std::path::Path;

/// 30 ms @ 16 kHz — Silero v4 frame size.
pub const FRAME_SAMPLES: usize = 480;

/// Default threshold in voiced frames (180 ms cumulative) below which the input
/// is treated as silence/noise and the caller should skip ASR.
/// FD-003 M3.5 #3: lowered from 8 → 6 to allow short commands through.
pub const DEFAULT_MIN_VOICE_FRAMES: usize = 6;

/// Default Silero confidence threshold per frame.
/// FD-003 M3.5 #3: raised from 0.3 → 0.6 to reduce noise mis-triggers.
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.6;

// Keep the old names as aliases so existing callers don't break.
pub const MIN_VOICE_FRAMES: usize = DEFAULT_MIN_VOICE_FRAMES;
pub const VAD_THRESHOLD: f32 = DEFAULT_VAD_THRESHOLD;

/// Configuration for the silence gate.
///
/// Parameterising the thresholds makes unit testing (and future M4 tuning)
/// possible without touching production constants.  The default impl yields
/// the same behaviour as the hard-coded values that shipped in M0.
#[derive(Debug, Clone, Copy)]
pub struct SilenceGateConfig {
    /// Minimum number of voiced frames required to classify audio as speech.
    /// Below this the gate returns [`SilenceGate::Silence`].
    /// Default: [`DEFAULT_MIN_VOICE_FRAMES`] (= 8 frames = 240 ms @ 30 ms/frame).
    pub min_voice_frames: usize,

    /// Per-frame Silero confidence threshold.  Frames with a score below this
    /// value are counted as silent.
    /// Default: [`DEFAULT_VAD_THRESHOLD`] (= 0.3).
    pub vad_threshold: f32,
}

impl Default for SilenceGateConfig {
    fn default() -> Self {
        Self {
            min_voice_frames: DEFAULT_MIN_VOICE_FRAMES,
            vad_threshold: DEFAULT_VAD_THRESHOLD,
        }
    }
}

/// Outcome of a silence-gate check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilenceGate {
    /// VAD ran successfully and the audio has enough voiced frames.
    Speech {
        voice_frames: usize,
        total_frames: usize,
    },
    /// VAD ran successfully and the audio is silence/noise.
    Silence {
        voice_frames: usize,
        total_frames: usize,
    },
    /// VAD could not be initialised (model missing, etc.). Caller should
    /// fail open and proceed with the engine.
    Unavailable,
}

/// Run Silero VAD over `audio` (16 kHz mono f32) using the model at
/// `vad_model_path` and decide whether the clip contains speech.
///
/// Uses [`SilenceGateConfig::default`] thresholds.  For custom thresholds
/// call [`check_with_config`] instead.
///
/// Returns `Unavailable` rather than `Err` when the model cannot be loaded —
/// the gate is a performance/correctness optimisation, not a hard dependency.
pub fn check(audio: &[f32], vad_model_path: &Path) -> SilenceGate {
    check_with_config(audio, vad_model_path, SilenceGateConfig::default())
}

/// Run Silero VAD over `audio` (16 kHz mono f32) using the model at
/// `vad_model_path` and decide whether the clip contains speech, using the
/// supplied [`SilenceGateConfig`].
///
/// Returns `Unavailable` rather than `Err` when the model cannot be loaded.
pub fn check_with_config(
    audio: &[f32],
    vad_model_path: &Path,
    config: SilenceGateConfig,
) -> SilenceGate {
    let mut vad = match SileroVad::new(vad_model_path, config.vad_threshold) {
        Ok(v) => v,
        Err(_) => return SilenceGate::Unavailable,
    };

    let total_frames = audio.len() / FRAME_SAMPLES;
    let voice_frames = audio
        .chunks(FRAME_SAMPLES)
        .filter(|f| f.len() == FRAME_SAMPLES)
        .filter(|f| vad.is_voice(f).unwrap_or(true))
        .count();

    if voice_frames < config.min_voice_frames {
        SilenceGate::Silence {
            voice_frames,
            total_frames,
        }
    } else {
        SilenceGate::Speech {
            voice_frames,
            total_frames,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────
//
// Tests that require an actual Silero model file are not run here (they would
// need the onnx resource in the test binary's resource path).  Instead we
// exercise the logic layer:
//
//   • The `check_with_config` helper's *decision logic* is tested via the
//     `decide_from_frame_counts` helper below (a pure function extracted for
//     testability that mirrors the if/else in check_with_config).
//
//   • Model-absent behaviour is tested via `check` / `check_with_config` with
//     a nonexistent path (should return Unavailable).
//
// The SileroVad integration is covered by the end-to-end silence_gate call
// in do_transcribe (SenseVoice branch) which has a real model at runtime.

/// Pure decision function used in tests and mirrored in check_with_config.
pub fn decide_from_frame_counts(
    voice_frames: usize,
    total_frames: usize,
    min_voice_frames: usize,
) -> SilenceGate {
    if voice_frames < min_voice_frames {
        SilenceGate::Silence {
            voice_frames,
            total_frames,
        }
    } else {
        SilenceGate::Speech {
            voice_frames,
            total_frames,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unavailable when model is missing ────────────────────────────────────

    #[test]
    fn unavailable_when_model_missing() {
        let audio = vec![0.0_f32; FRAME_SAMPLES * 100];
        let outcome = check(&audio, Path::new("/nonexistent/silero.onnx"));
        assert_eq!(outcome, SilenceGate::Unavailable);
    }

    #[test]
    fn unavailable_with_config_when_model_missing() {
        let audio = vec![0.0_f32; FRAME_SAMPLES * 100];
        let cfg = SilenceGateConfig::default();
        let outcome = check_with_config(&audio, Path::new("/nonexistent/silero.onnx"), cfg);
        assert_eq!(outcome, SilenceGate::Unavailable);
    }

    // ── Frame constant sanity ─────────────────────────────────────────────────

    #[test]
    fn frame_constants_match_silero_v4() {
        // Silero v4 expects 30 ms frames @ 16 kHz: 480 samples.
        assert_eq!(FRAME_SAMPLES, 480);
        // FD-003 M3.5 #3: DEFAULT_MIN_VOICE_FRAMES lowered from 8 → 6
        // (180 ms cumulative voiced audio).  6 × 30 ms = 180 ms, which
        // still rejects pure noise while admitting short one-word commands.
        assert_eq!(DEFAULT_MIN_VOICE_FRAMES, 6);
        assert_eq!(
            (DEFAULT_MIN_VOICE_FRAMES * FRAME_SAMPLES * 1000) / 16000,
            180
        );
    }

    #[test]
    fn default_config_matches_production_constants() {
        let cfg = SilenceGateConfig::default();
        assert_eq!(cfg.min_voice_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert!((cfg.vad_threshold - DEFAULT_VAD_THRESHOLD).abs() < f32::EPSILON);
    }

    // ── Decision logic tests (no model required) ──────────────────────────────

    /// Short audio < 0.3 s: 0 voice frames → Silence regardless of threshold
    #[test]
    fn short_audio_under_300ms_gated() {
        // 0.2 s @ 16 kHz = 3200 samples = 6 frames, 0 voiced → Silence
        let total_frames = 6usize;
        let voice_frames = 0usize; // all silent
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Silence {
                voice_frames,
                total_frames
            }
        );
    }

    /// Full silence 30 s: many frames, 0 voice → Silence
    #[test]
    fn full_silence_30s_gated() {
        // 30 s @ 16 kHz = 480_000 samples = 1000 frames, all silent
        let total_frames = 1000usize;
        let voice_frames = 0usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Silence {
                voice_frames,
                total_frames
            }
        );
    }

    /// Extremely weak signal (DC offset / near-zero RMS): voice_frames=0 → Silence
    #[test]
    fn extremely_weak_signal_gated() {
        // Simulated: VAD would see 0 frames as voiced for DC-offset / sub-threshold RMS.
        let total_frames = 50usize;
        let voice_frames = 0usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert!(matches!(result, SilenceGate::Silence { .. }));
    }

    /// High noise floor with signal below threshold: voice_frames < min → Silence
    #[test]
    fn high_noise_floor_below_threshold_gated() {
        // 5 voiced frames out of 100 total — just under the threshold of 6
        let total_frames = 100usize;
        let voice_frames = 5usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Silence {
                voice_frames,
                total_frames
            }
        );
    }

    /// Normal speech: voice_frames > threshold → Speech
    #[test]
    fn normal_speech_passes_gate() {
        // Typical 2s utterance: ~67 frames, ~30 voiced
        let total_frames = 67usize;
        let voice_frames = 30usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Speech {
                voice_frames,
                total_frames
            }
        );
    }

    /// Boundary case: exactly at threshold → Speech (>= is the pass condition)
    #[test]
    fn exactly_at_threshold_passes_gate() {
        let total_frames = 20usize;
        let voice_frames = DEFAULT_MIN_VOICE_FRAMES; // exactly 6 (FD-003 M3.5)
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Speech {
                voice_frames,
                total_frames
            },
            "voice_frames == min_voice_frames should be Speech (not gated)"
        );
    }

    /// Boundary case: one below threshold → Silence
    #[test]
    fn one_below_threshold_gated() {
        let total_frames = 20usize;
        let voice_frames = DEFAULT_MIN_VOICE_FRAMES - 1; // exactly 5 (FD-003 M3.5)
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Silence {
                voice_frames,
                total_frames
            },
            "voice_frames == min_voice_frames - 1 should be Silence"
        );
    }

    // ── Custom config tests ───────────────────────────────────────────────────

    #[test]
    fn custom_config_lower_threshold_passes_short_speech() {
        // With min_voice_frames=1, even a single voiced frame passes.
        let cfg = SilenceGateConfig {
            min_voice_frames: 1,
            vad_threshold: DEFAULT_VAD_THRESHOLD,
        };
        let total_frames = 10usize;
        let voice_frames = 1usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, cfg.min_voice_frames);
        assert!(matches!(result, SilenceGate::Speech { .. }));
    }

    #[test]
    fn custom_config_higher_threshold_rejects_sparse_speech() {
        // With min_voice_frames=20, 8 voiced frames is not enough.
        let cfg = SilenceGateConfig {
            min_voice_frames: 20,
            vad_threshold: DEFAULT_VAD_THRESHOLD,
        };
        let total_frames = 100usize;
        let voice_frames = 8usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, cfg.min_voice_frames);
        assert!(matches!(result, SilenceGate::Silence { .. }));
    }

    // ── Empty audio ───────────────────────────────────────────────────────────

    #[test]
    fn empty_audio_returns_silence() {
        // 0 samples → 0 frames → 0 voice_frames < 8 → Silence
        let total_frames = 0usize;
        let voice_frames = 0usize;
        let result = decide_from_frame_counts(voice_frames, total_frames, DEFAULT_MIN_VOICE_FRAMES);
        assert_eq!(
            result,
            SilenceGate::Silence {
                voice_frames: 0,
                total_frames: 0
            }
        );
    }
}
