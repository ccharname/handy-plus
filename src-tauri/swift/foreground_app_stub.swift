import Foundation

// Stub implementation — foreground app bridge is only compiled for macOS.
// This file exists as a fallback in case the build script selects it.

@_cdecl("get_foreground_app")
public func getForegroundApp() -> UnsafeMutablePointer<ForegroundAppInfo>? {
    return nil
}

@_cdecl("free_foreground_app_info")
public func freeForegroundAppInfo(_ info: UnsafeMutablePointer<ForegroundAppInfo>?) {
    // nothing to free
}
