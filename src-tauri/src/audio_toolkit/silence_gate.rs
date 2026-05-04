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

/// Threshold in voiced frames (240 ms cumulative) below which the input is
/// treated as silence/noise and the caller should skip ASR.
pub const MIN_VOICE_FRAMES: usize = 8;

/// Silero confidence threshold per frame.
pub const VAD_THRESHOLD: f32 = 0.3;

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
/// Returns `Unavailable` rather than `Err` when the model cannot be loaded —
/// the gate is a performance/correctness optimisation, not a hard dependency.
pub fn check(audio: &[f32], vad_model_path: &Path) -> SilenceGate {
    let mut vad = match SileroVad::new(vad_model_path, VAD_THRESHOLD) {
        Ok(v) => v,
        Err(_) => return SilenceGate::Unavailable,
    };

    let total_frames = audio.len() / FRAME_SAMPLES;
    let voice_frames = audio
        .chunks(FRAME_SAMPLES)
        .filter(|f| f.len() == FRAME_SAMPLES)
        .filter(|f| vad.is_voice(f).unwrap_or(true))
        .count();

    if voice_frames < MIN_VOICE_FRAMES {
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

    #[test]
    fn unavailable_when_model_missing() {
        let audio = vec![0.0_f32; FRAME_SAMPLES * 100];
        let outcome = check(&audio, Path::new("/nonexistent/silero.onnx"));
        assert_eq!(outcome, SilenceGate::Unavailable);
    }

    #[test]
    fn frame_constants_match_silero_v4() {
        // Silero v4 expects 30 ms frames @ 16 kHz: 480 samples.
        assert_eq!(FRAME_SAMPLES, 480);
        // 240 ms = 8 × 30 ms — short enough to allow legitimate one-word
        // commands (~300 ms) through, long enough to reject pure silence.
        assert_eq!(MIN_VOICE_FRAMES, 8);
        assert_eq!((MIN_VOICE_FRAMES * FRAME_SAMPLES * 1000) / 16000, 240);
    }
}
