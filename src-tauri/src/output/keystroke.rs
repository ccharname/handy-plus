//! Keystroke-based output sink.
//!
//! Simulates key presses via `enigo` to type text directly into the focused
//! application.  `enigo.text()` on macOS uses `CGEventCreateKeyboardEvent`
//! with a Unicode UTF-16 payload (`CGEventKeyboardSetUnicodeString`), which
//! bypasses the active input source's compose buffer entirely. CJK IMEs
//! therefore do not intercept the events and we do not need to switch
//! keyboard layouts.
//!
//! Earlier versions of this sink wrapped each `append()` in an
//! `InputSourceGuard` (TIS API) to force ABC layout. That caused
//! `EXC_BREAKPOINT` (SIGTRAP, PAC trap) on macOS 26 when called from a
//! worker thread — the Text Services Manager refuses cross-thread access
//! under pointer authentication. The guard is removed; the `ime` module is
//! retained as dead code for archaeology in case a future use case really
//! needs main-thread-only TIS access.
//!
//! This sink is the right choice for Electron apps (Cursor, VSCode, Slack,
//! Discord, web browsers) where the Accessibility API does not reach inside
//! Chromium's custom rendering layer.

use enigo::{Enigo, Keyboard, Settings};
use log::{debug, warn};

use super::StreamingSink;

/// Keystroke-based streaming sink.
pub struct KeystrokeOutput {
    enigo: Enigo,
}

impl KeystrokeOutput {
    /// Create a new `KeystrokeOutput`.
    ///
    /// Returns an error if `Enigo` cannot be initialised (e.g. missing
    /// Accessibility permissions on macOS).
    pub fn new() -> Result<Self, String> {
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| format!("Failed to initialise Enigo for KeystrokeOutput: {}", e))?;
        Ok(Self { enigo })
    }

    fn do_type(&mut self, text: &str) -> Result<(), String> {
        self.enigo
            .text(text)
            .map_err(|e| format!("Enigo text() failed: {}", e))
    }
}

impl StreamingSink for KeystrokeOutput {
    fn append(&mut self, delta: &str) -> Result<(), String> {
        if delta.is_empty() {
            return Ok(());
        }

        match self.do_type(delta) {
            Ok(()) => {
                debug!("[keystroke] typed {} chars", delta.chars().count());
                Ok(())
            }
            Err(e) => {
                warn!("[keystroke] typing failed: {}", e);
                Err(e)
            }
        }
    }

    fn finalize(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn cancel(&mut self) {
        // cancel = stop, NOT undo. enigo.text() is non-blocking from our
        // perspective; nothing to tear down.
        debug!("[keystroke] cancelled");
    }

    fn kind_str(&self) -> &'static str {
        "keystroke"
    }
}
