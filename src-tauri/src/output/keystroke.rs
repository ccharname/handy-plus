//! Keystroke-based output sink.
//!
//! Simulates key presses via `enigo` to type text directly into the focused
//! application.  On macOS, an [`InputSourceGuard`] is activated on the first
//! `append()` call to switch the keyboard to ABC (US) layout, preventing
//! Chinese/Japanese IME composition buffers from intercepting the events.
//! The guard is dropped on `cancel()` or `finalize()`, restoring the original
//! input source.
//!
//! This sink is best for Electron apps (Cursor, VSCode, Slack, Discord, web
//! browsers) where the Accessibility API does not reach inside Chromium's
//! custom rendering layer.

use enigo::{Enigo, Keyboard, Settings};
use log::{debug, warn};

use super::StreamingSink;

#[cfg(target_os = "macos")]
use super::ime::InputSourceGuard;

/// Keystroke-based streaming sink.
pub struct KeystrokeOutput {
    enigo: Enigo,
    #[cfg(target_os = "macos")]
    ime_guard: Option<InputSourceGuard>,
}

impl KeystrokeOutput {
    /// Create a new `KeystrokeOutput`.
    ///
    /// Returns an error if `Enigo` cannot be initialised (e.g. missing
    /// Accessibility permissions on macOS).
    pub fn new() -> Result<Self, String> {
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| format!("Failed to initialise Enigo for KeystrokeOutput: {}", e))?;
        Ok(Self {
            enigo,
            #[cfg(target_os = "macos")]
            ime_guard: None,
        })
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

        // On macOS, activate the IME guard lazily on the first append.
        // This switches to ABC layout so key events are not intercepted by IME.
        #[cfg(target_os = "macos")]
        if self.ime_guard.is_none() {
            self.ime_guard = Some(InputSourceGuard::new_abc());
            // Small delay to let the input source switch take effect before typing.
            std::thread::sleep(std::time::Duration::from_millis(30));
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
        // Drop IME guard — restores original input source.
        #[cfg(target_os = "macos")]
        {
            self.ime_guard = None;
        }
        Ok(())
    }

    fn cancel(&mut self) {
        // cancel = stop, NOT undo.
        // Drop the IME guard to restore input source immediately.
        #[cfg(target_os = "macos")]
        {
            self.ime_guard = None;
        }
        debug!("[keystroke] cancelled; IME guard released");
    }

    fn kind_str(&self) -> &'static str {
        "keystroke"
    }
}
