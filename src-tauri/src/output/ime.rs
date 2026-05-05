//! IME input-source guard for macOS.
//!
//! When typing via `enigo` key events, the system routes keystrokes through
//! the current input method.  If the user has a Chinese IME active, individual
//! key events are intercepted by the IME composition buffer and never reach the
//! target application as literal characters.
//!
//! This module provides [`InputSourceGuard`]: an RAII guard that switches to
//! the ABC (US) keyboard layout on construction and restores the original
//! input source on drop.  This ensures that `enigo::key_sequence()` calls
//! bypass IME composition while the guard is live.
//!
//! # Platform
//!
//! Only compiled on macOS.  All other platforms get a no-op stub.

// ─────────────────────────────────────────────────────────────────────────────
// macOS implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
pub use macos::InputSourceGuard;

#[cfg(target_os = "macos")]
mod macos {
    use log::{debug, warn};
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_void};

    // ── Carbon TIS types + functions (publicly linked on macOS) ──────────────

    // TISInputSourceRef is an opaque pointer to the Carbon Text Input Source.
    // We declare it as `*mut c_void` — that's what the C header typedef resolves to
    // when stripped of Objective-C sugar.
    #[allow(non_camel_case_types)]
    type TISInputSourceRef = *mut c_void;

    // CFStringRef — an opaque pointer to a Core Foundation string object.
    #[allow(non_camel_case_types)]
    type CFStringRef = *const c_void;

    // CFArrayRef — an opaque pointer to a Core Foundation array object.
    #[allow(non_camel_case_types)]
    type CFArrayRef = *const c_void;

    extern "C" {
        // Returns the currently selected keyboard input source.
        fn TISCopyCurrentKeyboardInputSource() -> TISInputSourceRef;

        // Copies an input source property value.  Returns a CFTypeRef (void*).
        fn TISGetInputSourceProperty(source: TISInputSourceRef, prop: CFStringRef) -> *mut c_void;

        // Selects (activates) an input source.
        fn TISSelectInputSource(source: TISInputSourceRef);

        // Creates a list of input sources matching a filter dictionary (NULL = all).
        fn TISCreateInputSourceList(
            filter: *const c_void,
            include_all_installed: bool,
        ) -> CFArrayRef;

        // CFRelease — decrements the retain count of any CF object.
        fn CFRelease(obj: *const c_void);

        // CFArrayGetCount — returns the number of elements.
        fn CFArrayGetCount(arr: CFArrayRef) -> isize;

        // CFArrayGetValueAtIndex — returns the element at index.
        fn CFArrayGetValueAtIndex(arr: CFArrayRef, index: isize) -> *const c_void;

        // CFStringGetCString — copies CF string content into a C buffer.
        fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: isize,
            encoding: u32,
        ) -> bool;
    }

    // kCFStringEncodingUTF8 = 0x08000100
    const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    // kTISPropertyInputSourceID — a CFString constant.
    //
    // We use a static CStr to get a *const c_char that we cast to CFStringRef.
    // The actual symbol is exported from the Carbon framework with the C name
    // `kTISPropertyInputSourceID`.  Apple's TIS headers use extern const for it.
    extern "C" {
        static kTISPropertyInputSourceID: *const c_void;
    }

    /// Extract the input-source ID string from a TISInputSourceRef.
    ///
    /// Returns the CFBundleIdentifier-like string, e.g.
    /// "com.apple.keylayout.ABC" or "com.apple.inputmethod.SCIM".
    ///
    /// # Safety
    /// `source` must be a non-null, live TISInputSourceRef.
    unsafe fn source_id(source: TISInputSourceRef) -> Option<String> {
        if source.is_null() {
            return None;
        }
        let prop = unsafe { kTISPropertyInputSourceID };
        let cf_str = unsafe { TISGetInputSourceProperty(source, prop) };
        if cf_str.is_null() {
            return None;
        }
        let mut buf = [0u8; 256];
        let ok = unsafe {
            CFStringGetCString(
                cf_str as CFStringRef,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as isize,
                CF_STRING_ENCODING_UTF8,
            )
        };
        if ok {
            unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
                .to_str()
                .ok()
                .map(|s| s.to_owned())
        } else {
            None
        }
    }

    /// Find the ABC (US English) keyboard input source from the installed list.
    ///
    /// Looks for the canonical bundle ID "com.apple.keylayout.ABC".
    /// Falls back to the first source whose ID contains "ABC" or "USInternational".
    unsafe fn find_abc_source() -> Option<TISInputSourceRef> {
        let all = unsafe { TISCreateInputSourceList(std::ptr::null(), false) };
        if all.is_null() {
            return None;
        }
        let count = unsafe { CFArrayGetCount(all) };
        let mut found: Option<TISInputSourceRef> = None;

        for i in 0..count {
            let src = unsafe { CFArrayGetValueAtIndex(all, i) } as TISInputSourceRef;
            if let Some(id) = unsafe { source_id(src) } {
                if id == "com.apple.keylayout.ABC"
                    || id == "com.apple.keylayout.US"
                    || id.contains("keylayout.ABC")
                {
                    found = Some(src);
                    break;
                }
            }
        }

        // Don't release individual elements — they are owned by the array.
        unsafe { CFRelease(all as *const c_void) };
        found
    }

    /// RAII guard: switch to ABC (US) keyboard input source on construction,
    /// restore the original source on drop.
    ///
    /// This is necessary for `KeystrokeOutput` on macOS because:
    /// - Chinese / Japanese IMEs intercept individual key events via their
    ///   composition buffer before they reach the target application.
    /// - Switching to a direct Latin layout ensures each key event is delivered
    ///   as a literal Unicode character.
    ///
    /// The guard is a no-op when the user already has ABC selected.
    pub struct InputSourceGuard {
        /// The input source that was active when the guard was created, or
        /// `None` if we didn't switch (ABC was already active).
        saved: Option<TISInputSourceRef>,
    }

    impl InputSourceGuard {
        /// Create a guard, switching to ABC if needed.
        ///
        /// Safe to call even if TIS APIs are unavailable — it degrades silently
        /// to a no-op guard.
        pub fn new_abc() -> Self {
            let saved = unsafe { Self::maybe_switch_to_abc() };
            Self { saved }
        }

        /// Attempt to switch to ABC keyboard layout.
        ///
        /// Returns the saved source (to restore later) if a switch was made,
        /// or `None` if no switch was needed / possible.
        unsafe fn maybe_switch_to_abc() -> Option<TISInputSourceRef> {
            let current = unsafe { TISCopyCurrentKeyboardInputSource() };
            if current.is_null() {
                return None;
            }

            let current_id = unsafe { source_id(current) };
            debug!("[ime] current input source: {:?}", current_id);

            let is_abc = current_id
                .as_deref()
                .map(|id| id.contains("keylayout.ABC") || id.contains("keylayout.US"))
                .unwrap_or(false);

            if is_abc {
                // Already ABC — no switch needed, release the copy.
                unsafe { CFRelease(current) };
                return None;
            }

            // Find and select ABC.
            if let Some(abc) = unsafe { find_abc_source() } {
                debug!("[ime] switching to ABC: {:?}", unsafe { source_id(abc) });
                unsafe { TISSelectInputSource(abc) };
                // Return the saved (current) source so we can restore it on drop.
                Some(current)
            } else {
                warn!("[ime] could not locate ABC keyboard layout; IME guard is a no-op");
                unsafe { CFRelease(current) };
                None
            }
        }
    }

    impl Drop for InputSourceGuard {
        fn drop(&mut self) {
            if let Some(saved) = self.saved.take() {
                let id = unsafe { source_id(saved) };
                debug!("[ime] restoring input source: {:?}", id);
                unsafe {
                    TISSelectInputSource(saved);
                    CFRelease(saved);
                }
            }
        }
    }

    // SAFETY: TISInputSourceRef is a CF opaque pointer managed by the Carbon
    // framework.  We only hold it alive within the guard's lifetime (from
    // `new_abc()` to `drop()`).  Since `InputSourceGuard` is only created and
    // dropped on the same thread (the output sink thread), and we never send it
    // across threads, marking it Send is safe in practice.  The Carbon TIS API
    // is not thread-safe in general, but select/copy operations on the main
    // thread (or within a single Tokio task) are fine.
    unsafe impl Send for InputSourceGuard {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-macOS stub
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
pub use stub::InputSourceGuard;

#[cfg(not(target_os = "macos"))]
mod stub {
    /// No-op IME guard for non-macOS platforms.
    pub struct InputSourceGuard;

    impl InputSourceGuard {
        pub fn new_abc() -> Self {
            InputSourceGuard
        }
    }
}
