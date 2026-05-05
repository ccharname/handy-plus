//! Accessibility-based output sink for macOS.
//!
//! Uses the macOS Accessibility API (AXUIElement) to directly set the value
//! of the focused text field.  This bypasses the keyboard event system
//! entirely, which means IME composition buffers never intercept the text.
//!
//! # Strategy
//!
//! 1. Get the system-wide AXUIElement.
//! 2. Ask for `kAXFocusedUIElementAttribute` to find the focused control.
//! 3. Read `kAXValueAttribute` (current text).
//! 4. Append the delta and write it back via `AXUIElementSetAttributeValue`.
//!
//! This "read → append → write" approach is the simplest and most compatible.
//! A future improvement would use `kAXSelectedTextRangeAttribute` to insert at
//! the actual cursor position rather than always appending to the field value.
//!
//! # Limitations
//!
//! - Requires Accessibility permissions (System Preferences → Security & Privacy).
//! - Not all controls expose `kAXValueAttribute` as writable (e.g. web content
//!   in Chrome/Electron uses a different accessibility layer).  The sink returns
//!   `Err` for unsupported elements; the caller should fall back to `KeystrokeOutput`.
//! - Only compiled on macOS.

#[cfg(target_os = "macos")]
pub use macos::AccessibilityOutput;

#[cfg(target_os = "macos")]
mod macos {
    use log::{debug, warn};
    use std::os::raw::c_void;

    use super::super::StreamingSink;

    // ── AX types ─────────────────────────────────────────────────────────────

    #[allow(non_camel_case_types)]
    type AXUIElementRef = *mut c_void;

    #[allow(non_camel_case_types)]
    type CFTypeRef = *mut c_void;

    #[allow(non_camel_case_types)]
    type AXError = i32;

    #[allow(non_camel_case_types)]
    type CFStringRef = *const c_void;

    // AXError constants
    const AX_SUCCESS: AXError = 0;

    // CFString encoding
    const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        fn AXUIElementSetAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: CFTypeRef,
        ) -> AXError;
        fn CFRelease(obj: *const c_void);
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const std::os::raw::c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut std::os::raw::c_char,
            buffer_size: isize,
            encoding: u32,
        ) -> bool;
        fn CFStringGetLength(the_string: CFStringRef) -> isize;
    }

    extern "C" {
        static kAXFocusedUIElementAttribute: *const c_void;
        static kAXValueAttribute: *const c_void;
    }

    /// Convert a Rust &str into a CFStringRef.  The caller is responsible for
    /// releasing the returned CFString via `CFRelease`.
    unsafe fn rust_str_to_cf(s: &str) -> CFStringRef {
        // Ensure null-terminated.
        let c_str = std::ffi::CString::new(s).unwrap_or_default();
        unsafe { CFStringCreateWithCString(std::ptr::null(), c_str.as_ptr(), CF_STRING_ENCODING_UTF8) }
    }

    /// Convert a CFStringRef to a Rust String.  Does NOT release `cf`.
    unsafe fn cf_to_rust_string(cf: CFStringRef) -> Option<String> {
        if cf.is_null() {
            return None;
        }
        let len = unsafe { CFStringGetLength(cf) };
        if len <= 0 {
            return Some(String::new());
        }
        // Allocate buffer with extra space for null terminator and multi-byte chars.
        let buf_size = (len as usize) * 4 + 1;
        let mut buf = vec![0u8; buf_size];
        let ok = unsafe {
            CFStringGetCString(
                cf,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf_size as isize,
                CF_STRING_ENCODING_UTF8,
            )
        };
        if ok {
            let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as *const std::os::raw::c_char) };
            s.to_str().ok().map(|s| s.to_owned())
        } else {
            None
        }
    }

    /// Accessibility-based text output.
    ///
    /// Best for native macOS apps (Notes, TextEdit, Pages, Mail, Safari).
    /// Falls back gracefully when the focused element does not support
    /// `kAXValueAttribute` writes.
    pub struct AccessibilityOutput {
        /// Accumulated text appended so far (for finalize).
        buffer: String,
    }

    impl AccessibilityOutput {
        pub fn new() -> Self {
            Self {
                buffer: String::new(),
            }
        }

        /// Append `text` to the currently focused AX element.
        ///
        /// Strategy: read current value → concat delta → write back.
        fn ax_append(text: &str) -> Result<(), String> {
            if text.is_empty() {
                return Ok(());
            }
            unsafe {
                // 1. System-wide element.
                let system = AXUIElementCreateSystemWide();
                if system.is_null() {
                    return Err("AXUIElementCreateSystemWide returned null".into());
                }

                // 2. Focused element.
                let mut focused: CFTypeRef = std::ptr::null_mut();
                let err = AXUIElementCopyAttributeValue(
                    system,
                    kAXFocusedUIElementAttribute as CFStringRef,
                    &mut focused,
                );
                CFRelease(system as *const c_void);

                if err != AX_SUCCESS || focused.is_null() {
                    return Err(format!("Could not get focused element (AXError {})", err));
                }

                // 3. Read current value.
                let mut current_val: CFTypeRef = std::ptr::null_mut();
                let err2 = AXUIElementCopyAttributeValue(
                    focused as AXUIElementRef,
                    kAXValueAttribute as CFStringRef,
                    &mut current_val,
                );

                let current_text = if err2 == AX_SUCCESS && !current_val.is_null() {
                    let t = cf_to_rust_string(current_val as CFStringRef).unwrap_or_default();
                    CFRelease(current_val);
                    t
                } else {
                    // Value not readable — might still be writable (some text fields
                    // don't expose a readable value but accept set).
                    String::new()
                };

                // 4. Write new value = current + delta.
                let new_text = format!("{}{}", current_text, text);
                let cf_new = rust_str_to_cf(&new_text);
                if cf_new.is_null() {
                    CFRelease(focused);
                    return Err("Failed to create CFString for new value".into());
                }

                let err3 = AXUIElementSetAttributeValue(
                    focused as AXUIElementRef,
                    kAXValueAttribute as CFStringRef,
                    cf_new as CFTypeRef,
                );
                CFRelease(cf_new as *const c_void);
                CFRelease(focused);

                if err3 != AX_SUCCESS {
                    return Err(format!(
                        "AXUIElementSetAttributeValue failed (AXError {})",
                        err3
                    ));
                }

                debug!("[accessibility] appended {} chars", text.chars().count());
                Ok(())
            }
        }
    }

    impl Default for AccessibilityOutput {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StreamingSink for AccessibilityOutput {
        fn append(&mut self, delta: &str) -> Result<(), String> {
            match Self::ax_append(delta) {
                Ok(()) => {
                    self.buffer.push_str(delta);
                    Ok(())
                }
                Err(e) => {
                    warn!("[accessibility] append failed: {}", e);
                    Err(e)
                }
            }
        }

        fn finalize(&mut self) -> Result<(), String> {
            // The streaming approach already wrote everything incrementally.
            // Nothing to do on finalize (the buffer tracks what was written).
            self.buffer.clear();
            Ok(())
        }

        fn cancel(&mut self) {
            // cancel = stop, NOT undo.  Already-written text stays.
            self.buffer.clear();
        }

        fn kind_str(&self) -> &'static str {
            "accessibility"
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-macOS stub (compilation only — never actually instantiated on non-macOS)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
pub use stub::AccessibilityOutput;

#[cfg(not(target_os = "macos"))]
mod stub {
    use super::super::StreamingSink;

    pub struct AccessibilityOutput;

    impl AccessibilityOutput {
        pub fn new() -> Self {
            Self
        }
    }

    impl Default for AccessibilityOutput {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StreamingSink for AccessibilityOutput {
        fn append(&mut self, _delta: &str) -> Result<(), String> {
            Err("AccessibilityOutput not available on this platform".into())
        }

        fn finalize(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn cancel(&mut self) {}

        fn kind_str(&self) -> &'static str {
            "accessibility"
        }
    }
}
