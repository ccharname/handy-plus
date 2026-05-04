fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_apple_intelligence_bridge();

    #[cfg(target_os = "macos")]
    build_apple_speech_bridge();

    #[cfg(target_os = "macos")]
    build_foreground_app_bridge();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_mlx_audio_bridge();

    generate_tray_translations();

    tauri_build::build()
}

/// Generate tray menu translations from frontend locale files.
///
/// Source of truth: src/i18n/locales/*/translation.json
/// The English "tray" section defines the struct fields.
fn generate_tray_translations() {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let locales_dir = Path::new("../src/i18n/locales");

    println!("cargo:rerun-if-changed=../src/i18n/locales");

    // Collect all locale translations
    let mut translations: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for entry in fs::read_dir(locales_dir).unwrap().flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let lang = path.file_name().unwrap().to_str().unwrap().to_string();
        let json_path = path.join("translation.json");

        println!("cargo:rerun-if-changed={}", json_path.display());

        let content = fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        if let Some(tray) = parsed.get("tray").cloned() {
            translations.insert(lang, tray);
        }
    }

    // English defines the schema
    let english = translations.get("en").unwrap().as_object().unwrap();
    let fields: Vec<_> = english
        .keys()
        .map(|k| (camel_to_snake(k), k.clone()))
        .collect();

    // Generate code
    let mut out = String::from(
        "// Auto-generated from src/i18n/locales/*/translation.json - do not edit\n\n",
    );

    // Struct
    out.push_str("#[derive(Debug, Clone)]\npub struct TrayStrings {\n");
    for (rust_field, _) in &fields {
        out.push_str(&format!("    pub {rust_field}: String,\n"));
    }
    out.push_str("}\n\n");

    // Static map
    out.push_str(
        "pub static TRANSLATIONS: Lazy<HashMap<&'static str, TrayStrings>> = Lazy::new(|| {\n",
    );
    out.push_str("    let mut m = HashMap::new();\n");

    for (lang, tray) in &translations {
        out.push_str(&format!("    m.insert(\"{lang}\", TrayStrings {{\n"));
        for (rust_field, json_key) in &fields {
            let val = tray.get(json_key).and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!(
                "        {rust_field}: \"{}\".to_string(),\n",
                escape_string(val)
            ));
        }
        out.push_str("    });\n");
    }

    out.push_str("    m\n});\n");

    fs::write(Path::new(&out_dir).join("tray_translations.rs"), out).unwrap();

    println!(
        "cargo:warning=Generated tray translations: {} languages, {} fields",
        translations.len(),
        fields.len()
    );
}

fn camel_to_snake(s: &str) -> String {
    s.chars()
        .enumerate()
        .fold(String::new(), |mut acc, (i, c)| {
            if c.is_uppercase() && i > 0 {
                acc.push('_');
            }
            acc.push(c.to_lowercase().next().unwrap());
            acc
        })
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_apple_intelligence_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const REAL_SWIFT_FILE: &str = "swift/apple_intelligence.swift";
    const STUB_SWIFT_FILE: &str = "swift/apple_intelligence_stub.swift";
    const BRIDGE_HEADER: &str = "swift/apple_intelligence_bridge.h";

    println!("cargo:rerun-if-changed={REAL_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={STUB_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={BRIDGE_HEADER}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("apple_intelligence.o");
    let static_lib_path = out_dir.join("libapple_intelligence.a");

    // SDKROOT/SWIFTC env-var overrides let non-Xcode toolchains (e.g. nixpkgs
    // with apple-sdk_* + standalone swift) bypass xcrun, which is Xcode-only.
    let sdk_path = env::var("SDKROOT").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .expect("Failed to locate macOS SDK")
                .stdout,
        )
        .expect("SDK path is not valid UTF-8")
        .trim()
        .to_string()
    });

    // Check if the SDK supports FoundationModels (required for Apple Intelligence)
    let framework_path =
        Path::new(&sdk_path).join("System/Library/Frameworks/FoundationModels.framework");
    let has_foundation_models = framework_path.exists();

    let source_file = if has_foundation_models {
        println!("cargo:warning=Building with Apple Intelligence support.");
        REAL_SWIFT_FILE
    } else {
        println!("cargo:warning=Apple Intelligence SDK not found. Building with stubs.");
        STUB_SWIFT_FILE
    };

    if !Path::new(source_file).exists() {
        panic!("Source file {} is missing!", source_file);
    }

    // See SDKROOT note above — same env-override pattern for non-Xcode toolchains.
    let swiftc_path = env::var("SWIFTC").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--find", "swiftc"])
                .output()
                .expect("Failed to locate swiftc")
                .stdout,
        )
        .expect("swiftc path is not valid UTF-8")
        .trim()
        .to_string()
    });

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Use macOS 11.0 as deployment target for compatibility
    // The @available(macOS 26.0, *) checks in Swift handle runtime availability
    // Weak linking for FoundationModels is handled via cargo:rustc-link-arg below
    let status = Command::new(&swiftc_path)
        .args([
            // Without this flag swiftc treats single-file input as script
            // mode and emits its own `_main` symbol into the .o, which can
            // win the link against Rust's main under some linkers (e.g.
            // open-source ld64 used in nixpkgs' Darwin stdenv), producing a
            // binary whose main() is a 5-instruction no-op that returns 0.
            // `-parse-as-library` keeps the compilation in library mode so
            // no `_main` is emitted. See:
            //   https://forums.swift.org/t/main-in-a-single-swift-file/63079
            "-parse-as-library",
            "-target",
            "arm64-apple-macosx11.0",
            "-sdk",
            &sdk_path,
            "-O",
            "-import-objc-header",
            BRIDGE_HEADER,
            "-c",
            source_file,
            "-o",
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to invoke swiftc for Apple Intelligence bridge");

    if !status.success() {
        panic!("swiftc failed to compile {source_file}");
    }

    let status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path
                .to_str()
                .expect("Failed to convert static lib path to string"),
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to create static library for Apple Intelligence bridge");

    if !status.success() {
        panic!("libtool failed for Apple Intelligence bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apple_intelligence");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());
    println!("cargo:rustc-link-lib=framework=Foundation");

    if has_foundation_models {
        // Use weak linking so the app can launch on systems without FoundationModels
        println!("cargo:rustc-link-arg=-weak_framework");
        println!("cargo:rustc-link-arg=FoundationModels");
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

#[cfg(target_os = "macos")]
fn build_apple_speech_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const REAL_SWIFT_FILE: &str = "swift/apple_speech.swift";
    const STUB_SWIFT_FILE: &str = "swift/apple_speech_stub.swift";
    const BRIDGE_HEADER: &str = "swift/apple_speech_bridge.h";

    println!("cargo:rerun-if-changed={REAL_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={STUB_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={BRIDGE_HEADER}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("apple_speech.o");
    let static_lib_path = out_dir.join("libapple_speech.a");

    // SDKROOT/SWIFTC env-var overrides let non-Xcode toolchains bypass xcrun.
    let sdk_path = env::var("SDKROOT").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .expect("Failed to locate macOS SDK")
                .stdout,
        )
        .expect("SDK path is not valid UTF-8")
        .trim()
        .to_string()
    });

    // SFSpeechRecognizer is available from macOS 10.15 on both architectures.
    // Always use the real implementation on macOS; the stub is for non-macOS only.
    let source_file = if Path::new(REAL_SWIFT_FILE).exists() {
        println!("cargo:warning=Building with Apple Speech (SFSpeechRecognizer) support.");
        REAL_SWIFT_FILE
    } else {
        println!("cargo:warning=apple_speech.swift not found. Building with stub.");
        STUB_SWIFT_FILE
    };

    if !Path::new(source_file).exists() {
        panic!("Source file {} is missing!", source_file);
    }

    let swiftc_path = env::var("SWIFTC").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--find", "swiftc"])
                .output()
                .expect("Failed to locate swiftc")
                .stdout,
        )
        .expect("swiftc path is not valid UTF-8")
        .trim()
        .to_string()
    });

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Determine the triple based on target architecture.
    // SFSpeechRecognizer is available on both aarch64 and x86_64.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let swift_target = if target_arch == "aarch64" {
        "arm64-apple-macosx11.0"
    } else {
        "x86_64-apple-macosx11.0"
    };

    let status = Command::new(&swiftc_path)
        .args([
            "-parse-as-library",
            "-target",
            swift_target,
            "-sdk",
            &sdk_path,
            "-O",
            "-import-objc-header",
            BRIDGE_HEADER,
            "-c",
            source_file,
            "-o",
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to invoke swiftc for Apple Speech bridge");

    if !status.success() {
        panic!("swiftc failed to compile {source_file}");
    }

    let status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path
                .to_str()
                .expect("Failed to convert static lib path to string"),
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to create static library for Apple Speech bridge");

    if !status.success() {
        panic!("libtool failed for Apple Speech bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apple_speech");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());
    println!("cargo:rustc-link-lib=framework=Foundation");
    // Speech and AVFoundation are available since macOS 10.15, no weak linking needed
    println!("cargo:rustc-link-lib=framework=Speech");
    println!("cargo:rustc-link-lib=framework=AVFoundation");

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

#[cfg(target_os = "macos")]
fn build_foreground_app_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const REAL_SWIFT_FILE: &str = "swift/foreground_app.swift";
    const STUB_SWIFT_FILE: &str = "swift/foreground_app_stub.swift";
    const BRIDGE_HEADER: &str = "swift/foreground_app_bridge.h";

    println!("cargo:rerun-if-changed={REAL_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={STUB_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={BRIDGE_HEADER}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("foreground_app.o");
    let static_lib_path = out_dir.join("libforeground_app.a");

    // SDKROOT/SWIFTC env-var overrides let non-Xcode toolchains bypass xcrun.
    let sdk_path = env::var("SDKROOT").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .expect("Failed to locate macOS SDK")
                .stdout,
        )
        .expect("SDK path is not valid UTF-8")
        .trim()
        .to_string()
    });

    // NSWorkspace (AppKit) and CGWindowListCopyWindowInfo (CoreGraphics) are
    // available since macOS 10.13 on both architectures — always use real impl.
    let source_file = if Path::new(REAL_SWIFT_FILE).exists() {
        println!("cargo:warning=Building with foreground app detection support.");
        REAL_SWIFT_FILE
    } else {
        println!("cargo:warning=foreground_app.swift not found. Building with stub.");
        STUB_SWIFT_FILE
    };

    if !Path::new(source_file).exists() {
        panic!("Source file {} is missing!", source_file);
    }

    let swiftc_path = env::var("SWIFTC").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--find", "swiftc"])
                .output()
                .expect("Failed to locate swiftc")
                .stdout,
        )
        .expect("swiftc path is not valid UTF-8")
        .trim()
        .to_string()
    });

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Support both aarch64 and x86_64 (same as apple_speech_bridge).
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let swift_target = if target_arch == "aarch64" {
        "arm64-apple-macosx11.0"
    } else {
        "x86_64-apple-macosx11.0"
    };

    let status = Command::new(&swiftc_path)
        .args([
            "-parse-as-library",
            "-target",
            swift_target,
            "-sdk",
            &sdk_path,
            "-O",
            "-import-objc-header",
            BRIDGE_HEADER,
            "-c",
            source_file,
            "-o",
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to invoke swiftc for foreground app bridge");

    if !status.success() {
        panic!("swiftc failed to compile {source_file}");
    }

    let status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path
                .to_str()
                .expect("Failed to convert static lib path to string"),
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to create static library for foreground app bridge");

    if !status.success() {
        panic!("libtool failed for foreground app bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=foreground_app");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());
    println!("cargo:rustc-link-lib=framework=Foundation");
    // AppKit and CoreGraphics are always available on macOS — no weak linking needed.
    println!("cargo:rustc-link-lib=framework=AppKit");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

/// Build the mlx-audio-swift bridge for Apple Silicon macOS.
///
/// Strategy:
///   1. Run `swift build -c release --target MLXAudioBridge` on the subpackage.
///   2. Collect ALL .o files from the SPM build directory (mlx-audio-swift pulls ~15 packages).
///   3. Merge with `libtool -static -filelist` into libmlx_audio.a.
///   4. Emit cargo:rustc-link-* directives for the merged lib + required Apple frameworks.
///
/// Deployment target: macosx14.0 — MLX requires macOS 14 for the Metal 3 shader APIs.
/// This is gated on aarch64 only; x86_64 Macs cannot run MLX.
///
/// SPM mirror config: if the dev machine has a pre-resolved .build (from the PoC),
/// SPM will reuse it. On first run it clones mlx-audio-swift + 14 transitive deps.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_mlx_audio_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const PKG_PATH: &str = "mlx_bridge_pkg";

    println!("cargo:rerun-if-changed={PKG_PATH}/Package.swift");
    println!("cargo:rerun-if-changed={PKG_PATH}/Sources/MLXAudioBridge/bridge.swift");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let static_lib_path = out_dir.join("libmlx_audio.a");

    // Locate SDK and swiftc via xcrun.
    let sdk_path = env::var("SDKROOT").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .expect("Failed to locate macOS SDK")
                .stdout,
        )
        .expect("SDK path is not valid UTF-8")
        .trim()
        .to_string()
    });

    let swiftc_path = env::var("SWIFTC").unwrap_or_else(|_| {
        String::from_utf8(
            Command::new("xcrun")
                .args(["--find", "swiftc"])
                .output()
                .expect("Failed to locate swiftc")
                .stdout,
        )
        .expect("swiftc path is not valid UTF-8")
        .trim()
        .to_string()
    });

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Derive the absolute package path relative to the manifest directory.
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let pkg_abs = manifest_dir.join(PKG_PATH);

    println!("cargo:warning=Building mlx-audio-swift bridge (SPM resolve + compile — first run may take several minutes)");

    // Step 1: `swift build -c release --target MLXAudioBridge`
    // SDKROOT must be set so SPM picks the right macOS SDK.
    // Deployment target 14.0 — required by MLX Metal backend.
    // SWIFT_DEPLOYMENT_TARGET tells swiftc to emit macosx14.0 min-version load commands.
    let build_status = Command::new("swift")
        .args([
            "build",
            "-c",
            "release",
            "--target",
            "MLXAudioBridge",
            "--package-path",
            pkg_abs.to_str().expect("pkg path"),
        ])
        .env("SDKROOT", &sdk_path)
        .env("SWIFT_DEPLOYMENT_TARGET", "14.0")
        .status()
        .expect("Failed to invoke `swift build` for MLXAudioBridge");

    if !build_status.success() {
        panic!("swift build failed for MLXAudioBridge. Check SPM resolve / network access.");
    }

    // Step 2: collect all .o files produced by SPM (includes mlx-audio-swift's deps).
    // SPM places them under <pkg>/.build/arm64-apple-macosx/release/
    let build_dir = pkg_abs
        .join(".build")
        .join("arm64-apple-macosx")
        .join("release");

    let mut object_files: Vec<PathBuf> = Vec::new();
    collect_object_files(&build_dir, &mut object_files);

    if object_files.is_empty() {
        panic!(
            "No .o files found under {}. swift build may have succeeded without producing objects.",
            build_dir.display()
        );
    }

    println!(
        "cargo:warning=MLXAudioBridge: merging {} .o files into libmlx_audio.a",
        object_files.len()
    );

    // Step 3: merge all objects into a single static lib with libtool.
    // libtool handles the whitespace-in-filename issue that plagues filenames like
    //   "OrderedDictionary+Partial MutableCollection.swift.o"
    // by accepting a file-list argument via -filelist.
    let filelist_path = out_dir.join("mlx_audio_objects.txt");
    let filelist_content = object_files
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&filelist_path, filelist_content)
        .expect("Failed to write object file list");

    let libtool_status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path.to_str().expect("static lib path"),
            "-filelist",
            filelist_path.to_str().expect("filelist path"),
        ])
        .status()
        .expect("Failed to invoke libtool for MLXAudioBridge");

    if !libtool_status.success() {
        panic!("libtool failed to merge .o files into libmlx_audio.a");
    }

    // Step 4: emit Cargo link directives.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mlx_audio");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());

    // Required Apple frameworks for MLX Metal + audio pipeline.
    // Foundation: Swift runtime / strings / concurrency.
    // Metal / MetalPerformanceShaders / MetalPerformanceShadersGraph: MLX GPU backend.
    // Accelerate: MLX CPU fallback + BLAS.
    // CoreML: optional CoreML execution provider (may be used by some mlx-audio-swift paths).
    // AVFoundation: audio file loading (loadAudioArray uses AVAudioFile internally).
    for framework in &[
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
        "Accelerate",
        "CoreML",
        "AVFoundation",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }

    // Swift runtime libs that bridge.swift uses transitively. Without explicit
    // link directives, libswift_Concurrency / libswiftFoundation are dropped
    // by the linker (Cargo doesn't autolink Swift libs the way swiftc does)
    // and `Task { ... }` calls become silent no-ops at runtime — Tasks never
    // execute and any `await` blocks the calling thread forever. The pthread
    // bridge in bridge.swift::runSync depends on Concurrency to schedule the
    // detached task that drives mlx-audio-swift's async API.
    for swift_lib in &[
        "swift_Concurrency",
        "swiftFoundation",
        "swiftCore",
    ] {
        println!("cargo:rustc-link-lib={swift_lib}");
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

/// Recursively collect all `.o` files under `dir` into `files`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn collect_object_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_object_files(&path, files);
        } else if path.extension().and_then(|e| e.to_str()) == Some("o") {
            files.push(path);
        }
    }
}
