//! Sink routing: decide which output sink to use based on the frontmost app.
//!
//! # Decision table
//!
//! | App bundle ID pattern | Sink |
//! |---|---|
//! | `com.apple.Notes` / `TextEdit` / `Pages` / `Mail` / `Safari` / most native macOS apps | `AccessibilityOutput` |
//! | `com.todesktop.*` (Cursor) / `com.microsoft.VSCode` / Electron apps / Slack / Discord | `KeystrokeOutput` + IME guard |
//! | Unknown / fallback | `KeystrokeOutput` first; error → `ClipboardPasteOutput` |
//!
//! The `active_frontmost_app_bundle_id()` function delegates to
//! `crate::foreground::current_foreground_app()`, which already has the
//! Swift bridge for NSWorkspace on macOS.

use log::{debug, info, warn};

use super::{
    accessibility::AccessibilityOutput, clipboard::ClipboardPasteOutput,
    keystroke::KeystrokeOutput, StreamingSink,
};

/// Get the bundle ID of the frontmost application, if available.
///
/// Returns `None` on non-macOS platforms or when no app is frontmost.
pub fn active_frontmost_app_bundle_id() -> Option<String> {
    crate::foreground::current_foreground_app()
        .and_then(|app| app.bundle_id)
}

/// Categorise a bundle ID into a sink preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SinkPreference {
    Accessibility,
    Keystroke,
}

fn classify_bundle_id(bundle_id: &str) -> SinkPreference {
    // Native macOS apps that expose kAXValueAttribute on their text fields.
    let accessibility_patterns = [
        "com.apple.Notes",
        "com.apple.TextEdit",
        "com.apple.Pages",
        "com.apple.mail",
        "com.apple.Safari",
        "com.apple.Keynote",
        "com.apple.Numbers",
        "com.apple.AddressBook",
        "com.apple.iWork",
        "com.apple.Calendar",
        "org.libreoffice",
        "com.microsoft.Word",
        "com.microsoft.Outlook",
        "com.apple.finder",
        "com.apple.ScriptEditor",
        "com.apple.systempreferences",
    ];

    // Electron and web-based apps — Accessibility API won't reach their web
    // content layer.
    let keystroke_patterns = [
        "com.todesktop.",          // Cursor AI editor
        "com.microsoft.VSCode",
        "com.microsoft.vscode",
        "dev.zed.",                // Zed editor
        "com.github.atom",
        "com.sublimetext.",
        "com.jetbrains.",
        "io.alacritty",
        "net.kovidgoyal.kitty",
        "com.googlecode.iterm2",
        "com.apple.Terminal",
        "org.gnu.Emacs",
        "io.brackets.",
        "com.slack.",
        "com.tinyspeck.slackmacgap",
        "com.hnc.Discord",
        "org.whispersystems.signal-desktop",
        "com.electron.",
        ".electron.",
        "electron",
    ];

    let bundle_lower = bundle_id.to_lowercase();

    for pat in &accessibility_patterns {
        if bundle_id.starts_with(pat) || bundle_id == *pat {
            return SinkPreference::Accessibility;
        }
    }

    for pat in &keystroke_patterns {
        if bundle_lower.contains(pat) {
            return SinkPreference::Keystroke;
        }
    }

    // Default: prefer Keystroke (handles more edge cases gracefully).
    SinkPreference::Keystroke
}

/// Select the best output sink for the given active app bundle ID.
///
/// # Fallback chain
///
/// 1. If `active_app` maps to `Accessibility`, try `AccessibilityOutput`.
///    On failure, fall back to `KeystrokeOutput`, then `ClipboardPasteOutput`.
/// 2. If `active_app` maps to `Keystroke`, try `KeystrokeOutput`.
///    On failure (e.g. Accessibility permissions not granted), fall back to
///    `ClipboardPasteOutput`.
/// 3. `ClipboardPasteOutput` is always available as the last resort.
pub fn select_sink(active_app: Option<&str>) -> Box<dyn StreamingSink + Send> {
    let bundle_id = active_app.unwrap_or("");

    let preference = if bundle_id.is_empty() {
        debug!("[routing] no active app detected; defaulting to Keystroke");
        SinkPreference::Keystroke
    } else {
        let pref = classify_bundle_id(bundle_id);
        info!(
            "[routing] bundle_id={:?} → preference={:?}",
            bundle_id, pref
        );
        pref
    };

    match preference {
        SinkPreference::Accessibility => {
            // Try Accessibility first.
            // We don't have a cheap "is_supported" check without a real element,
            // so we create it and let the first append() surface any failure.
            #[cfg(target_os = "macos")]
            {
                info!("[routing] selected AccessibilityOutput");
                Box::new(AccessibilityOutput::new())
            }
            #[cfg(not(target_os = "macos"))]
            {
                // Non-macOS: fall through to Keystroke.
                select_keystroke_or_clipboard()
            }
        }
        SinkPreference::Keystroke => select_keystroke_or_clipboard(),
    }
}

fn select_keystroke_or_clipboard() -> Box<dyn StreamingSink + Send> {
    match KeystrokeOutput::new() {
        Ok(sink) => {
            info!("[routing] selected KeystrokeOutput");
            Box::new(sink)
        }
        Err(e) => {
            warn!(
                "[routing] KeystrokeOutput unavailable ({}); falling back to ClipboardPasteOutput",
                e
            );
            Box::new(ClipboardPasteOutput::new())
        }
    }
}

/// Convenience: query frontmost app and select the best sink in one call.
pub fn select_sink_auto() -> Box<dyn StreamingSink + Send> {
    let bundle_id = active_frontmost_app_bundle_id();
    select_sink(bundle_id.as_deref())
}
