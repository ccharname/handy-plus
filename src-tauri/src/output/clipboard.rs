//! Clipboard-based one-shot output sink.
//!
//! This is the safest fallback when Accessibility API and keystrokes both fail.
//! It is **not** truly streaming — all `append()` calls accumulate text in an
//! internal buffer, and `finalize()` performs a single paste (write clipboard →
//! Cmd+V → restore original clipboard).
//!
//! **User clipboard preservation**: the current clipboard content is saved
//! before the paste and restored after a short delay (100 ms) so that the
//! user's copy history is not polluted.

use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use log::{debug, info, warn};

use super::StreamingSink;

/// One-shot clipboard paste sink.
///
/// All `append()` calls are buffered; `finalize()` flushes the buffer in a
/// single clipboard write + Cmd+V.
pub struct ClipboardPasteOutput {
    buffer: String,
}

impl ClipboardPasteOutput {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Write `text` to the system clipboard, paste via Cmd+V, then restore
    /// the original clipboard content after 100 ms.
    fn paste_via_clipboard(text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }

        info!("[clipboard] one-shot paste ({} chars)", text.chars().count());

        // Save current clipboard content.
        let original = read_clipboard().unwrap_or_default();

        // Write new text to clipboard.
        write_clipboard(text)?;

        // Small delay to ensure clipboard write has propagated.
        std::thread::sleep(std::time::Duration::from_millis(30));

        // Send Cmd+V (macOS) or Ctrl+V (Windows/Linux).
        let mut enigo = Enigo::new(&Settings::default())
            .map_err(|e| format!("Enigo init failed for clipboard paste: {}", e))?;

        send_paste_key(&mut enigo)?;

        // Wait for the paste to complete before restoring.
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Restore original clipboard.
        if let Err(e) = write_clipboard(&original) {
            warn!("[clipboard] failed to restore original clipboard: {}", e);
        }

        debug!("[clipboard] paste complete, clipboard restored");
        Ok(())
    }
}

impl Default for ClipboardPasteOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingSink for ClipboardPasteOutput {
    fn append(&mut self, delta: &str) -> Result<(), String> {
        // Buffer all deltas — we flush on finalize.
        self.buffer.push_str(delta);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), String> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let text = std::mem::take(&mut self.buffer);
        Self::paste_via_clipboard(&text)
    }

    fn cancel(&mut self) {
        // Discard buffer — do not paste.
        self.buffer.clear();
        debug!("[clipboard] sink cancelled; buffer discarded");
    }

    fn kind_str(&self) -> &'static str {
        "clipboard"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform-specific clipboard helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Read the current clipboard text content.  Returns `None` on failure.
fn read_clipboard() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("pbpaste").output().ok()?;
        String::from_utf8(output.stdout).ok()
    }
    #[cfg(target_os = "windows")]
    {
        // On Windows we'd use the `clipboard` crate or win32 API.
        // For now return None (non-critical — original content not restored, acceptable).
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        // Linux: try xclip / wl-paste.
        use std::process::Command;
        Command::new("wl-paste")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .or_else(|| {
                Command::new("xclip")
                    .args(["-selection", "clipboard", "-o"])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
            })
    }
}

/// Write `text` to the system clipboard.
fn write_clipboard(text: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("pbcopy")
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("pbcopy spawn failed: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| format!("pbcopy write failed: {}", e))?;
        }
        child
            .wait()
            .map_err(|e| format!("pbcopy wait failed: {}", e))?;
        Ok(())
    }
    #[cfg(target_os = "windows")]
    {
        // Stub — Windows clipboard write via enigo or win32 in a future iteration.
        let _ = text;
        Err("Clipboard write not implemented on Windows in ClipboardPasteOutput".into())
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // Try wl-copy (Wayland) first, then xclip (X11).
        let result_wl = Command::new("wl-copy")
            .stdin(Stdio::piped())
            .spawn()
            .ok()
            .and_then(|mut c| {
                c.stdin.take()?.write_all(text.as_bytes()).ok()?;
                c.wait().ok()
            });

        if result_wl.is_some() {
            return Ok(());
        }

        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("xclip spawn failed: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| format!("xclip write failed: {}", e))?;
        }
        child
            .wait()
            .map_err(|e| format!("xclip wait failed: {}", e))?;
        Ok(())
    }
}

/// Send the platform-appropriate paste key combination.
fn send_paste_key(enigo: &mut Enigo) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        // Cmd+V on macOS.
        enigo
            .key(Key::Meta, Direction::Press)
            .map_err(|e| format!("Meta press failed: {}", e))?;
        enigo
            .key(Key::Other(9), Direction::Click) // keycode 9 = 'v' on macOS
            .map_err(|e| format!("V click failed: {}", e))?;
        enigo
            .key(Key::Meta, Direction::Release)
            .map_err(|e| format!("Meta release failed: {}", e))?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Ctrl+V on Windows/Linux.
        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| format!("Ctrl press failed: {}", e))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| format!("V click failed: {}", e))?;
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| format!("Ctrl release failed: {}", e))?;
    }
    Ok(())
}
