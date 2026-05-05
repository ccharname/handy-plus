//! M2.5 — True streaming sink architecture (StreamingSink + IME-safe paste).
//!
//! This module provides the output layer for streaming ASR tokens directly
//! into the user's cursor position, matching the "native typing" experience.
//!
//! # Architecture
//!
//! ```text
//! Inference token callback
//!         │
//!         ▼
//!   DeltaComputer          ← conservative prefix-only delta extraction
//!         │ Action::Append(delta)
//!         ▼
//!   select_sink(app)       ← routing by frontmost app bundle ID
//!         │
//!   ┌─────┴──────────────────────────┐
//!   │ AccessibilityOutput (macOS)    │  ← AXUIElement value set; best for
//!   │                                │    native macOS apps
//!   ├────────────────────────────────┤
//!   │ KeystrokeOutput                │  ← enigo key_sequence + IME guard;
//!   │                                │    Electron / VSCode / Slack
//!   ├────────────────────────────────┤
//!   │ ClipboardPasteOutput (fallback)│  ← one-shot pbcopy + Cmd+V;
//!   │                                │    never streams (finalizes only)
//!   └────────────────────────────────┘
//! ```
//!
//! # Observability fields added by this module
//!
//! - `sink_kind` — "accessibility" | "keystroke" | "clipboard"
//! - `paste_lag_ms` — time from token arrival to screen update (per partial)
//!
//! These are emitted in T7Output stage records and feed the M5 SLA assertions.

pub mod accessibility;
pub mod clipboard;
pub mod delta;
// `ime` is retained as dead code: TIS API on macOS 26 SIGTRAPs (PAC trap)
// when accessed from worker threads. KeystrokeOutput now relies on
// `enigo.text()` (CGEvent Unicode payload) which bypasses IME compose
// buffers without needing input-source switching. Module kept for future
// reference if main-thread-only TIS access becomes necessary.
#[allow(dead_code)]
pub mod ime;
pub mod keystroke;
pub mod routing;

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A streaming output sink.
///
/// Each implementation targets a different mechanism for writing text into the
/// active application:
/// - [`accessibility::AccessibilityOutput`] — AXUIElement (macOS native apps)
/// - [`keystroke::KeystrokeOutput`] — enigo keystrokes (Electron / cross-platform)
/// - [`clipboard::ClipboardPasteOutput`] — clipboard one-shot (universal fallback)
///
/// # Threading
///
/// Sinks are always created and used on the same thread (the transcription
/// worker thread), so `Send` is sufficient — `Sync` is not required.
pub trait StreamingSink: Send {
    /// Append a text delta to the target application's input position.
    ///
    /// Called for each new streaming token from the ASR engine.
    /// May be called multiple times per utterance.
    ///
    /// Returns `Err` if the append failed (e.g. element not writable).
    fn append(&mut self, delta: &str) -> Result<(), String>;

    /// Finalize the current utterance.
    ///
    /// Called once at the end of inference.  For streaming sinks this is
    /// typically a no-op (text was already written incrementally).  For
    /// [`clipboard::ClipboardPasteOutput`] this triggers the actual paste.
    fn finalize(&mut self) -> Result<(), String>;

    /// Cancel the current output session.
    ///
    /// **Cancel = stop, NOT undo.**  Text already pasted to the target
    /// application is left as-is.  Any pending buffer is discarded.
    /// IME guards are released immediately.
    fn cancel(&mut self);

    /// Return a human-readable identifier for this sink type.
    ///
    /// Used in observability records under the `sink_kind` field.
    /// Values: `"accessibility"` | `"keystroke"` | `"clipboard"`.
    fn kind_str(&self) -> &'static str;
}

// ─────────────────────────────────────────────────────────────────────────────
// Observability field names (used in T7Output serde_json records)
// ─────────────────────────────────────────────────────────────────────────────

/// Observability field: sink kind tag.
/// Value: "accessibility" | "keystroke" | "clipboard"
pub const OBS_FIELD_SINK_KIND: &str = "sink_kind";

/// Observability field: time from partial token arrival to screen update (ms).
/// Measured per `append()` call; p50 ≤ 80 ms, p99 ≤ 200 ms (M5 SLA).
pub const OBS_FIELD_PASTE_LAG_MS: &str = "paste_lag_ms";

// ─────────────────────────────────────────────────────────────────────────────
// Public convenience re-exports
// ─────────────────────────────────────────────────────────────────────────────

pub use accessibility::AccessibilityOutput;
pub use clipboard::ClipboardPasteOutput;
pub use delta::{Action, DeltaComputer};
pub use keystroke::KeystrokeOutput;
pub use routing::{active_frontmost_app_bundle_id, select_sink, select_sink_auto};
