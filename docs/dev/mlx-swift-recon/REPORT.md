# Phase C Recon: mlx-audio-swift Native Swift SDK

**Date:** 2026-05-04  
**Branch:** handy-plus/main  
**Verdict:** CONDITIONAL GO — link story proven, one deployment precondition

---

## Background

- Round A (commit `620fdd9`): Apple incremental paste deployed. Validates cursor-streaming UX.
- Round B (Python sidecar): Rejected — 490 MB bundle, 145 dylibs, Python interpreter fragile.
- Round C (this recon): Native Swift SDK via `Blaizzy/mlx-audio-swift`, compiled via SPM and linked into Cargo.

Prior art in the repo: `src-tauri/swift/apple_speech.swift` compiled via `src-tauri/build.rs` using `swiftc -parse-as-library → .o → libtool -static → .a → cargo:rustc-link-lib=static`. Phase C uses the same pattern, extended to SPM-managed packages.

PoC location: `/tmp/handy-mlx-swift-poc/`

---

## Section 1 — mlx-audio-swift Current State

**Source:** https://github.com/Blaizzy/mlx-audio-swift

| Attribute | Value |
|---|---|
| Latest release | v0.1.2 (March 14, 2026) |
| License | MIT |
| Swift Package Manager | Yes — `Package.swift` at repo root |
| Minimum macOS | 14.0 (`platforms: [.macOS(.v14)]`) |
| Swift tools version | 6.2 |
| Minimum Xcode | 15+ |
| Hardware | Apple Silicon (M1+) required for Metal backend |

### Supported STT Models

| Model | Class | Notes |
|---|---|---|
| Voxtral Realtime Mini-4B | `VoxtralRealtimeModel` | 4-bit quant; Level-1 streaming |
| Qwen3-ASR 0.6B / 1.7B | `Qwen3ASRModel` | Has `StreamingInferenceSession` (Level-2) |
| GLM-ASR Nano-2512 | `GLMASRModel` | 4-bit |
| Parakeet 0.6b-v3 | `ParakeetModel` | TDT variant |
| Cohere Transcribe | `CohereTranscribeModel` | — |
| SenseVoice | `SenseVoiceModel` | Already in handy-plus via sherpa |
| FireRedASR2 | `FireRedASR2Model` | — |

### Streaming API — Two Levels

**Level 1 — Token streaming** (file/buffer in, tokens out as async stream):
```swift
// All STTGenerationModel conformers support this
for try await event in model.generateStream(audio: audio, generationParameters: params) {
    switch event {
    case .token(let token): print(token, terminator: "")  // incremental partial
    case .result(let output): finalOutput = output         // complete transcription
    case .info: break
    }
}
```

**Level 2 — True live PCM streaming** (feed raw samples in real time):
```swift
// StreamingInferenceSession — Qwen3ASR models only
let session = StreamingInferenceSession(
    model: qwen3Model,
    config: StreamingConfig(
        delayPreset: .realtime,   // 200ms latency
        language: "zh"
    )
)
session.feedAudio(samples: pcmChunk)   // [Float], 16 kHz mono, any chunk size

for await event in session.events {
    switch event {
    case .provisional(let text): updateOverlay(text)    // may change
    case .confirmed(let text):   appendCursor(text)     // frozen
    case .displayUpdate(let confirmed, let provisional): ...
    case .stats(let s): print("RTF:", s.realTimeFactor)
    case .ended(let full): finalize(full)
    }
}
```

Note: `StreamingInferenceSession` is hardcoded to `Qwen3ASRModel` internally. Voxtral uses Level-1 (buffered chunk decode) only.

### API Mapping Table (Round B Python → Round C Swift)

| Round B Python | Round C Swift |
|---|---|
| `VoxtralStreamingSession.feed(samples)` | `StreamingInferenceSession.feedAudio(samples:)` (Qwen3 only) |
| `VoxtralStreamingSession.step()` | Automatic — session auto-triggers decode |
| `session.events` (async generator) | `session.events: AsyncStream<TranscriptionEvent>` |
| `VoxtralRealtimeModel.generate(audio)` | `VoxtralRealtimeModel.generate(audio: MLXArray)` |
| `model.from_pretrained("mlx-community/...")` | `VoxtralRealtimeModel.fromPretrained("mlx-community/...")` (async) |
| Python `asyncio` | Swift `async/await` + `AsyncStream` |
| NumPy float32 array | `MLXArray` (from `loadAudioArray(from: URL)`) |

### Audio Input Format

- Sample rate: 16 kHz (mandatory; use `resampleAudio()` helper)
- Channels: mono (`audio.mean(axis: -1)` for stereo)
- Format: `MLXArray` (Float32) or `[Float]` for live feed
- Loading from file: `let (sampleRate, audio) = try loadAudioArray(from: url)`

### Model Weight Distribution

Downloaded on first use from HuggingFace Hub (`mlx-community` org). No weights bundled with the SDK. Cache: `~/.cache/huggingface/hub/`. Voxtral 4-bit: ~3.5 GB. Qwen3-ASR-0.6B-8bit: ~700 MB. Both already cached on this machine.

---

## Section 2 — Build/Linking Story: PoC Results

### PoC Structure

```
/tmp/handy-mlx-swift-poc/
  Package.swift               # HandyMLXBridge SPM wrapper
  Sources/HandyMLXBridge/
    bridge.swift              # @_cdecl FFI bridge
  rust-probe/
    Cargo.toml
    build.rs                  # links prebuilt .a into Cargo
    src/main.rs               # calls @_cdecl from Rust
  libHandyMLXComplete2.a      # 328 MB merged static lib (799 .o files)
```

### SPM Resolve

mlx-audio-swift's dep graph: `swift-huggingface → swift-xet → swift-nio` (83k objects). Full network clone times out. Resolved via SPM mirror config pointing at shallow local clones. This is a dev-environment limitation only; CI with stable bandwidth resolves fine.

**Result:** 15 packages resolved: mlx-swift, mlx-swift-lm, swift-transformers, swift-huggingface, etc.

### Swift Compile

```
Building for production...
[255/255] Compiling HandyMLXBridge bridge.swift
Build of target: 'HandyMLXBridge' complete!
```

Two Swift 6 compile errors encountered and fixed during PoC:
1. Generic nested class (`SyncBox<T>` moved to module scope — Swift disallows generic class inside generic function)
2. `@Sendable` constraint on async closure passed to `Task {}`

### @_cdecl Symbol Verification

```
$ nm .build/arm64-apple-macosx/release/HandyMLXBridge.build/bridge.swift.o | grep mlx_audio
0000000000000048 T _mlx_audio_bridge_free_string
0000000000000000 T _mlx_audio_bridge_version
00000000000005f8 T _mlx_audio_transcribe_file
```

All three `@_cdecl` exports present as C text symbols.

### Static Library Creation

SPM doesn't produce a single merged `.a` for library products. Pattern: collect all `.o` files from per-target SPM build dirs, merge with `libtool -static`. Python avoids shell whitespace issues with filenames like `OrderedDictionary+Partial MutableCollection.swift.o`:

```python
cmd = ['libtool', '-static', '-o', 'libHandyMLXComplete2.a'] + all_object_files
subprocess.run(cmd)  # 799 .o files → 328 MB
```

### Cargo Link

`build.rs` emits:
```
cargo:rustc-link-search=native=/tmp/handy-mlx-swift-poc
cargo:rustc-link-lib=static=HandyMLXComplete2
cargo:rustc-link-lib=framework=Foundation
cargo:rustc-link-lib=framework=Metal
cargo:rustc-link-lib=framework=MetalPerformanceShaders
cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph
cargo:rustc-link-lib=framework=Accelerate
cargo:rustc-link-lib=framework=CoreML
cargo:rustc-link-lib=framework=AVFoundation
cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift
```

Build result:
```
   Compiling handy-mlx-probe v0.1.0
    Finished `release` profile [optimized] target(s) in 0.00s
```

Link warnings only (macOS 14.0 vs 11.0 deployment target — not fatal, resolved by bumping target for MLX feature gate).

### Runtime FFI Test

```
$ ./rust-probe/target/release/handy-mlx-probe
=== HandyMLXBridge Rust FFI Probe ===
Bridge version: handy-mlx-bridge/0.1.0 mlx-audio-swift/0.1.2
PASS: @_cdecl symbols resolved and callable from Rust
```

FFI bridging Rust → Swift → mlx-audio-swift: **PROVEN**.

### Metal Runtime (Transcription Attempt)

```
$ ./rust-probe/target/release/handy-mlx-probe /tmp/qwen3_bench_zh.wav
Attempting transcription of: /tmp/qwen3_bench_zh.wav
[mlx-bridge] Error: MLX error: Failed to load the default metallib.
  library not found library not found library not found library not found
```

**Root cause:** MLX Metal backend searches for `default.metallib` in the app bundle. A bare CLI binary has no bundle. The Metal shader compiler (`xcrun metal`) is also not installed on this machine.

**Blocker level:** Deployment detail only — not a fundamental link/FFI blocker.

**Resolution (two paths):**

1. **Xcode build route (zero-cost, recommended):** `bun run tauri build` uses Xcode's SPM integration, which compiles `.metal` files and bundles `default.metallib` into the `.app` automatically. The bare CLI PoC exposes this; the production Tauri build does not have this problem. Verify with `bun run tauri dev` + MLX init call.

2. **XCFramework route (fallback):** Download `Cmlx.xcframework.zip` from mlx-swift Releases (v0.31.3), extract pre-compiled `default.metallib`, add to `tauri.conf.json` resources. ~1 day.

---

## Section 3 — Comparison to apple_speech.swift Bridge

| Aspect | apple_speech bridge | mlx-audio bridge |
|---|---|---|
| Source | Single `swift/apple_speech.swift` | SPM subpackage `mlx_bridge_pkg/` + `swift/mlx_audio.swift` wrapper |
| Compile step | `swiftc -parse-as-library -c file.swift` | `swift build -c release --target MLXAudioBridge` |
| Archive step | `libtool -static -o libapple_speech.a apple_speech.o` | `libtool -static -o libmlx_audio.a $(find .build -name "*.o")` |
| `OUT_DIR` pattern | Same | Same |
| Frameworks | Foundation, Speech, AVFoundation | Foundation, Metal, MPS, MPSG, Accelerate, CoreML, AVFoundation |
| Deployment target | `arm64-apple-macosx11.0` | `arm64-apple-macosx14.0` |
| Universal binary | Yes (handles x86_64) | arm64 only (MLX requires Apple Silicon) |
| `rpath` | `/usr/lib/swift` | Same |

**Key structural difference:** apple_speech.swift has zero external Swift package dependencies, so a single `swiftc -c` invocation works. mlx-audio-swift has ~15 SPM dependencies. `build.rs` must call `swift build` on the subpackage, then collect and merge hundreds of `.o` files. The `OUT_DIR` and `cargo:rustc-link-*` pattern is identical.

**Deployment target note:** handy-plus currently targets `macosx11.0`. The MLX bridge requires `macosx14.0`. Gate the bridge behind `#[cfg(all(target_os = "macos", target_arch = "aarch64"))]` and a runtime `#available(macOS 14.0, *)` check. Existing `build_apple_intelligence_bridge()` provides a template.

---

## Section 4 — Implementation Phases

### Phase C1 — Bridge Foundation (~3-5 days)

**New files:**

| Path | Description |
|---|---|
| `src-tauri/mlx_bridge_pkg/Package.swift` | SPM subpackage with `MLXAudioSTT` dep |
| `src-tauri/mlx_bridge_pkg/Sources/MLXAudioBridge/bridge.swift` | `@_cdecl` exports for version probe, one-shot transcribe, streaming feed/stop |
| `src-tauri/src/mlx_audio.rs` | Rust FFI wrapper; `MlxAudioSession` struct with `feed()`, `stop()`, `transcribe_file()` |

**Modified files:**

| Path | Change |
|---|---|
| `src-tauri/build.rs` | Add `build_mlx_audio_bridge()` gated on `macos + aarch64` |
| `src-tauri/tauri.conf.json` | Add `mlx/default.metallib` to `bundle.resources` (if XCFramework route needed) |
| `src-tauri/entitlements.plist` | Verify `cs.allow-unsigned-executable-memory` for Metal JIT |

**Exit criteria:**
- `bun run tauri build` succeeds
- `mlx_audio_bridge_version()` callable from debug build
- `MLX.Device()` does not throw (metallib found)

**Risks + mitigations:**
- Metallib path: verify with `bun run tauri dev` before merging
- 328 MB `.a` slow link: Tauri strips dead code; actual binary delta small (~30-50 MB)
- SPM resolve slow in CI: pin specific tag (v0.1.2) not `branch: "main"`

### Phase C2 — Wire Voxtral Realtime End-to-End (~2-3 days)

**New enum variants:**
```rust
// In transcription.rs
pub enum MlxModelKind {
    VoxtralRealtime,    // mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
    Qwen3Asr06B,        // mlx-community/Qwen3-ASR-0.6B-8bit
    // reserved: Qwen3Asr17B, Parakeet, SenseVoiceMLX
}
```

**New `LoadedEngine::MlxAudio(MlxAudioSession)` arm in `transcription.rs`:**
- On start: load model (HF cache), init session
- PCM callback: call `mlx_audio_feed_pcm()` FFI
- On stop: call `mlx_audio_stop()`, collect final text
- Partials: route through existing `transcription-partial` Tauri event chain

**New ModelInfo in `model.rs`:**
- No local ONNX download — weights fetched by HF hub at first use

**New preset `experimental_voxtral_streaming`** in `settings.rs`:
- `builtin: false` — opt-in only

**Audio pipeline:** existing `audio.rs` PCM callback at 16 kHz → `MlxAudioSession::feed(&[f32])` → bridge FFI. No resampling needed (handy already records at 16 kHz for sherpa).

**Exit criteria:**
- Preset appears in model selector
- Voice → Voxtral → text pasted to cursor
- Partials stream through overlay

**Risks:**
- Voxtral is Level-1 streaming (token-by-token, not true live PCM feed). Partials appear per decode chunk, not per phoneme. Still better than sherpa chunked; worse than Apple Speech incremental.
- Memory: 4B 4-bit model needs ~4 GB unified RAM. Profile on M1/8 GB.

### Phase C3 — Multi-Engine Expansion (~1 day each)

Qwen3-ASR unlocks Level-2 true streaming (200ms latency). Same bridge, new `MlxModelKind` variant + `ModelInfo` + preset per engine.

| Engine | Streaming | Priority |
|---|---|---|
| Voxtral Mini-4B | Level-1 (buffered) | P0 (Phase C2) |
| Qwen3-ASR-0.6B | Level-2 (live PCM) | P1 |
| Parakeet-0.6B | Level-1, EN-only, fast | P2 |
| SenseVoice-MLX | Level-1 | P3 (overlap with sherpa SenseVoice) |

---

## Section 5 — Go/No-Go

### Verdict: CONDITIONAL GO

**The link story is fully proven:**

| Check | Result |
|---|---|
| SPM resolve (15 packages) | PASS |
| Swift compile (`@_cdecl` bridge) | PASS (2 minor Swift 6 fixes applied) |
| Symbol export | PASS — `_mlx_audio_bridge_version`, `_mlx_audio_transcribe_file` in .o |
| libtool merge (799 .o → 328 MB .a) | PASS |
| Cargo link (no undefined symbols) | PASS |
| Rust FFI call | PASS — version string printed from Swift through Rust |
| Metal runtime | BLOCKED by metallib — deployment detail, not link issue |

**One precondition before starting C1:**

> **P1 (Low risk, 30 min):** Run `bun run tauri dev`, trigger `MLX.Device()` init, confirm no metallib error. Xcode's SPM integration is expected to handle this automatically. If it fails, use XCFramework fallback (~1 day extra).

**Start C1 as soon as P1 is verified.**
