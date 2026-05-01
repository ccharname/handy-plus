/// Information about the currently active (frontmost) application.
#[derive(Debug, Clone, serde::Serialize, specta::Type)]
pub struct ForegroundApp {
    /// macOS: CFBundleIdentifier (e.g. "com.apple.Safari")
    /// Windows: exe basename (e.g. "chrome.exe")
    /// Linux: always None (stub)
    pub bundle_id: Option<String>,
    /// Human-readable process/app name, cross-platform fallback.
    pub process_name: Option<String>,
    /// Title of the frontmost window.
    /// macOS: requires Screen Recording permission; None if not granted.
    /// Windows: from GetWindowTextW.
    /// Linux: always None (stub).
    pub window_title: Option<String>,
}

/// Return the current foreground application, or `None` if it cannot be determined.
pub fn current_foreground_app() -> Option<ForegroundApp> {
    #[cfg(target_os = "macos")]
    {
        macos::get()
    }
    #[cfg(target_os = "windows")]
    {
        windows::get()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

// ────────────────────────────────────────────────────────────────────────────
// macOS — Swift/ObjC bridge via libforeground_app.a
// ────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::CStr;
    use std::os::raw::c_char;

    use super::ForegroundApp;

    /// C struct layout must match `ForegroundAppInfo` in foreground_app_bridge.h
    #[repr(C)]
    struct ForegroundAppInfo {
        bundle_id: *mut c_char,
        process_name: *mut c_char,
        window_title: *mut c_char,
    }

    extern "C" {
        #[link_name = "get_foreground_app"]
        fn ffi_get_foreground_app() -> *mut ForegroundAppInfo;

        #[link_name = "free_foreground_app_info"]
        fn ffi_free_foreground_app_info(info: *mut ForegroundAppInfo);
    }

    /// Convert a nullable C string pointer to an `Option<String>`.
    /// # Safety
    /// `ptr` must be either null or a valid, NUL-terminated C string.
    unsafe fn cstr_to_option(ptr: *const c_char) -> Option<String> {
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }

    pub(super) fn get() -> Option<ForegroundApp> {
        let info_ptr = unsafe { ffi_get_foreground_app() };
        if info_ptr.is_null() {
            return None;
        }

        let info = unsafe { &*info_ptr };
        let app = ForegroundApp {
            bundle_id: unsafe { cstr_to_option(info.bundle_id) },
            process_name: unsafe { cstr_to_option(info.process_name) },
            window_title: unsafe { cstr_to_option(info.window_title) },
        };

        unsafe { ffi_free_foreground_app_info(info_ptr) };

        // Return Some even when all fields are None — caller can decide what to do
        // with an all-None struct (e.g. no frontmost app at login window).
        Some(app)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Windows — pure Rust via the `windows` crate
// ────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
mod windows {
    use super::ForegroundApp;

    use ::windows::core::PWSTR;
    use ::windows::Win32::Foundation::{CloseHandle, HWND};
    use ::windows::Win32::System::ProcessStatus::K32GetModuleBaseNameW;
    use ::windows::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
    };
    use ::windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId,
    };

    pub(super) fn get() -> Option<ForegroundApp> {
        // SAFETY: All Win32 calls here follow the documented usage patterns and
        // only operate on the current user session's window state.
        unsafe {
            let hwnd: HWND = GetForegroundWindow();
            if hwnd.0 == 0 {
                return None;
            }

            // Window title
            let mut title_buf = [0u16; 512];
            let title_len = GetWindowTextW(hwnd, &mut title_buf);
            let window_title = if title_len > 0 {
                Some(String::from_utf16_lossy(&title_buf[..title_len as usize]))
            } else {
                None
            };

            // Process ID
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid == 0 {
                return Some(ForegroundApp {
                    bundle_id: None,
                    process_name: None,
                    window_title,
                });
            }

            // Open process handle with minimal rights
            let process = match OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                false,
                pid,
            ) {
                Ok(h) => h,
                Err(_) => {
                    return Some(ForegroundApp {
                        bundle_id: None,
                        process_name: None,
                        window_title,
                    });
                }
            };

            // Get module (exe) basename
            let mut name_buf = [0u16; 260];
            let name_len = K32GetModuleBaseNameW(process, None, &mut name_buf);
            let _ = CloseHandle(process);

            let exe_name = if name_len > 0 {
                Some(String::from_utf16_lossy(&name_buf[..name_len as usize]))
            } else {
                None
            };

            Some(ForegroundApp {
                bundle_id: exe_name.clone(), // Windows has no bundle id; use exe basename
                process_name: exe_name,
                window_title,
            })
        }
    }
}
