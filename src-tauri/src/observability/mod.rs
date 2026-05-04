//! Full-pipeline observability for Handy.
//!
//! Provides 8-stage instrumentation across the hotkey→output pipeline with
//! structured JSONL logging, ULID-based request correlation, and per-stage
//! timing.  All writes go to `~/Library/Logs/Handy/handy.jsonl` (daily
//! rotation via `tracing-appender`).
//!
//! # Usage
//!
//! ```rust
//! use crate::observability::{RequestId, Stage, Outcome};
//! use crate::observability;
//!
//! let req = RequestId::new();
//! observability::record_stage(req, Stage::T0Hotkey, Outcome::Ok, 12.0, None);
//! ```
//!
//! For stages with extra metrics, pass a `serde_json::Value` map:
//!
//! ```rust
//! let meta = serde_json::json!({ "device_init_ms": 45, "capture_open_ms": 38 });
//! observability::record_stage(req, Stage::T1AudioCapture, Outcome::Ok, 55.0, Some(meta));
//! ```

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing_appender::non_blocking::WorkerGuard;
use ulid::Ulid;

// ── Public types ─────────────────────────────────────────────────────────────

/// Opaque request identifier — one per hotkey→output cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(Ulid);

impl RequestId {
    /// Generate a fresh monotonically-ordered request id.
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    pub fn as_str(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Owned(self.0.to_string())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Pipeline stages in order of execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// rdev keydown → coordinator dispatch
    T0Hotkey,
    /// cpal stream open → first frame received
    T1AudioCapture,
    /// Continuous recording duration metrics
    T2Recording,
    /// Silero VAD processing
    T3Vad,
    /// rubato 16kHz resample
    T4Resample,
    /// Model inference (SenseVoice or Qwen3-MLX)
    T5Inference,
    /// ITN / dedup / hotword injection
    T6Postprocess,
    /// Clipboard set + paste dispatch
    T7Output,
    /// Whole pipeline: t0 → t7
    Total,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::T0Hotkey => "t0_hotkey",
            Stage::T1AudioCapture => "t1_audio_capture",
            Stage::T2Recording => "t2_recording",
            Stage::T3Vad => "t3_vad",
            Stage::T4Resample => "t4_resample",
            Stage::T5Inference => "t5_inference",
            Stage::T6Postprocess => "t6_postprocess",
            Stage::T7Output => "t7_output",
            Stage::Total => "total",
        }
    }
}

/// Per-stage outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok,
    Cancelled,
    Error,
    Timeout,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Cancelled => "cancelled",
            Outcome::Error => "error",
            Outcome::Timeout => "timeout",
        }
    }
}

// ── Tracing guard (keeps the non-blocking writer alive) ───────────────────────

/// Held by `lib.rs` for the lifetime of the process.
pub struct ObservabilityGuard {
    _guard: WorkerGuard,
}

// ── Initialiser ───────────────────────────────────────────────────────────────

static LOG_TRANSCRIPTS: OnceCell<bool> = OnceCell::new();

/// Call once at startup (in `lib.rs` before any logging).
///
/// * `log_dir` — directory where `handy.jsonl` will be written (daily rotation)
/// * `log_transcripts` — if `true`, full transcript text is included in t6/t7
///   stage records; defaults to `false` (only char-count is emitted)
pub fn init(log_dir: std::path::PathBuf, log_transcripts: bool) -> ObservabilityGuard {
    LOG_TRANSCRIPTS.set(log_transcripts).unwrap_or_default();

    let file_appender = tracing_appender::rolling::daily(log_dir, "handy.jsonl");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    // Only install the subscriber if one is not already set (e.g. in tests).
    let _ = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with(
            fmt::layer()
                .json()
                .with_writer(non_blocking)
                // Each event → one line of JSON; no pretty-printing.
                .with_ansi(false),
        )
        .try_init();

    ObservabilityGuard { _guard: guard }
}

// ── Core emit helper ──────────────────────────────────────────────────────────

/// Emit one structured stage event.
///
/// Fields always present:
/// - `request_id`, `stage`, `outcome`, `duration_ms`, `ts_unix_ms`
///
/// `extra` may add stage-specific metrics (e.g. `rtf`, `vad_ms`, `cold`).
pub fn record_stage(
    req: RequestId,
    stage: Stage,
    outcome: Outcome,
    duration_ms: f64,
    extra: Option<serde_json::Value>,
) {
    // Merge extra fields into a flat object so the tracing event is compact.
    let extra_json = extra
        .as_ref()
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    let ts_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    tracing::info!(
        request_id = %req,
        stage = stage.as_str(),
        outcome = outcome.as_str(),
        duration_ms = duration_ms,
        ts_unix_ms = ts_unix_ms as u64,
        extra = %extra_json,
        "stage_event"
    );
}

/// Shorthand: record a successful stage without extra metadata.
#[inline]
pub fn ok(req: RequestId, stage: Stage, duration_ms: f64) {
    record_stage(req, stage, Outcome::Ok, duration_ms, None);
}

/// Shorthand: record a successful stage with extra metadata.
#[inline]
pub fn ok_with(req: RequestId, stage: Stage, duration_ms: f64, extra: serde_json::Value) {
    record_stage(req, stage, Outcome::Ok, duration_ms, Some(extra));
}

/// Shorthand: record a cancelled outcome.
#[inline]
pub fn cancelled(req: RequestId, stage: Stage, duration_ms: f64) {
    record_stage(req, stage, Outcome::Cancelled, duration_ms, None);
}

/// Shorthand: record an error outcome.
#[inline]
pub fn error(req: RequestId, stage: Stage, duration_ms: f64) {
    record_stage(req, stage, Outcome::Error, duration_ms, None);
}

/// Returns `true` when the caller should include raw transcript text in logs.
/// Defaults to `false`; overridden by `--log-transcripts` CLI flag.
pub fn log_transcripts() -> bool {
    *LOG_TRANSCRIPTS.get().unwrap_or(&false)
}

// ── Active-request state (Tauri managed state) ────────────────────────────────

/// Tauri managed state that carries the current in-flight request id from
/// `start()` into the async `stop()` pipeline.  Wrapped in a `Mutex` so it
/// can be set/read across threads.
pub struct ActiveRequestId(std::sync::Mutex<RequestId>);

impl ActiveRequestId {
    pub fn new() -> Self {
        Self(std::sync::Mutex::new(RequestId::new()))
    }

    pub fn set(&self, req: RequestId) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = req;
        }
    }

    pub fn get(&self) -> RequestId {
        self.0
            .lock()
            .map(|g| *g)
            .unwrap_or_else(|_| RequestId::new())
    }
}

impl Default for ActiveRequestId {
    fn default() -> Self {
        Self::new()
    }
}

// ── Convenience timer ─────────────────────────────────────────────────────────

/// Lightweight stopwatch with nanosecond precision.
pub struct Stopwatch(std::time::Instant);

impl Stopwatch {
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    /// Elapsed milliseconds as f64 (sub-ms precision).
    pub fn elapsed_ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}
