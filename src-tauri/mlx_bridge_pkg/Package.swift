// swift-tools-version:6.2
// MLXAudioBridge SPM subpackage for handy-plus.
//
// Compiled by src-tauri/build.rs via `swift build -c release --target MLXAudioBridge`.
// All @_cdecl exports are merged into libmlx_audio.a and linked into the Rust binary
// via cargo:rustc-link-lib=static=mlx_audio.
//
// Deployment target: macOS 14.0 — required by MLX Metal backend.
// Gate in build.rs: cfg(all(target_os = "macos", target_arch = "aarch64"))
// Apple Silicon only — MLX requires the Metal GPU backend.

import PackageDescription

let package = Package(
    name: "MLXAudioBridge",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "MLXAudioBridge",
            type: .static,
            targets: ["MLXAudioBridge"]
        )
    ],
    dependencies: [
        // Pin to v0.1.2 tag (not branch) for reproducible builds.
        // The PoC at /tmp/handy-mlx-swift-poc verified that this tag compiles
        // and all @_cdecl symbols export correctly.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", exact: "0.1.2")
    ],
    targets: [
        .target(
            name: "MLXAudioBridge",
            dependencies: [
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
            ],
            path: "Sources/MLXAudioBridge",
            swiftSettings: [
                // Swift 6 strict concurrency — bridge uses DispatchSemaphore pattern
                // so we can keep @Sendable at the module boundary without full actors.
                .swiftLanguageMode(.v6)
            ]
        )
    ]
)
