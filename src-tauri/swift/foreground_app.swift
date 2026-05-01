import AppKit
import CoreGraphics
import Foundation

// MARK: - Swift implementation for foreground app detection
// Compiled via Cargo build script for macOS targets (aarch64 + x86_64).
//
// Window title requires the Screen Recording permission (CGWindowListCopyWindowInfo
// returns kCGWindowName only when the caller has been granted that entitlement).
// If the permission is not granted we return nil for window_title rather than
// failing; the bundle_id and process_name fields are always populated when an
// app is frontmost.

private func dupCString(_ s: String?) -> UnsafeMutablePointer<CChar>? {
    guard let s = s else { return nil }
    return s.withCString { strdup($0) }
}

@_cdecl("get_foreground_app")
public func getForegroundApp() -> UnsafeMutablePointer<ForegroundAppInfo>? {
    let ptr = UnsafeMutablePointer<ForegroundAppInfo>.allocate(capacity: 1)
    ptr.initialize(to: ForegroundAppInfo(bundle_id: nil, process_name: nil, window_title: nil))

    // NSWorkspace gives us the frontmost app metadata without any special permissions.
    let workspace = NSWorkspace.shared
    guard let app = workspace.frontmostApplication else {
        // No frontmost app (e.g. during login window) — return empty struct.
        return ptr
    }

    ptr.pointee.bundle_id = dupCString(app.bundleIdentifier)
    ptr.pointee.process_name = dupCString(app.localizedName)

    // Window title via CGWindowListCopyWindowInfo.
    // This call silently returns kCGWindowName == nil (or omits the key entirely)
    // when Screen Recording permission has not been granted. We treat any absence
    // of kCGWindowName as "not available" rather than an error.
    let pid = app.processIdentifier
    let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
    if let windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] {
        for windowInfo in windowList {
            // kCGWindowLayer 0 is the normal application window layer.
            guard let layer = windowInfo[kCGWindowLayer as String] as? Int, layer == 0 else {
                continue
            }
            guard let ownerPID = windowInfo[kCGWindowOwnerPID as String] as? Int32,
                  ownerPID == pid else {
                continue
            }
            // kCGWindowName is absent (not just nil) when Screen Recording is denied.
            if let title = windowInfo[kCGWindowName as String] as? String {
                ptr.pointee.window_title = dupCString(title)
            }
            // Take only the first matching window.
            break
        }
    }

    return ptr
}

@_cdecl("free_foreground_app_info")
public func freeForegroundAppInfo(_ info: UnsafeMutablePointer<ForegroundAppInfo>?) {
    guard let info = info else { return }

    if let s = info.pointee.bundle_id {
        free(s)
    }
    if let s = info.pointee.process_name {
        free(s)
    }
    if let s = info.pointee.window_title {
        free(s)
    }

    info.deallocate()
}
