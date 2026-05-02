# Handy+ v0.8.7 Test Report

> Test date: PENDING | Platform: macOS / Apple M-series | Total cases: 67
> Pass / Fail / Skip: PENDING / PENDING / PENDING

## 1. Executive Summary

PENDING — to be filled after Wave 1-3 completes.

Key questions this report answers:
- Did v0.8.6→v0.8.7 fixes (Apple Speech permission hang, CT-Punc CJK fallback, bench latency inflation) actually land?
- What is the steady-state latency for each preset under real Tauri pipeline?
- Where are the remaining performance bottlenecks?

## 2. Coverage Matrix

| Category | Auto (A) | Semi-auto (SA) | Manual (H) | Total | Pass | Fail | Skip |
|---|---|---|---|---|---|---|---|
| ASR engines (T-1.x) | 9 | 2 | 0 | 11 | - | - | - |
| CT-Punc layer (T-2.x) | 7 | 0 | 0 | 7 | - | - | - |
| ITN Chinese (T-3.x) | 3 | 0 | 0 | 3 | - | - | - |
| 3-tier hotwords (T-4.x) | 3 | 0 | 1 | 4 | - | - | - |
| ASR Preset switch (T-5.x) | 2 | 0 | 2 | 4 | - | - | - |
| Power Mode profiles (T-6.x) | 6 | 1 | 0 | 7 | - | - | - |
| History retranscribe (T-7.x) | 1 | 0 | 2 | 3 | - | - | - |
| post_process_chain (T-8.x) | 3 | 0 | 1 | 4 | - | - | - |
| Diary archival (T-9.x) | 5 | 0 | 0 | 5 | - | - | - |
| Recording overlay (T-10.x) | 0 | 1 | 3 | 4 | - | - | - |
| Apple Intelligence (T-11.x) | 1 | 0 | 1 | 2 | - | - | - |
| Punc model UI (T-12.x) | 1 | 0 | 1 | 2 | - | - | - |
| Apple Speech permission (T-13.x) | 4 | 0 | 1 | 5 | - | - | - |
| Bench CLI (T-14.x) | 3 | 0 | 0 | 3 | - | - | - |
| Single-instance / CLI (T-15.x) | 3 | 0 | 1 | 4 | - | - | - |
| i18n (T-16.x) | 1 | 0 | 1 | 2 | - | - | - |
| VAD / Audio (T-17.x) | 3 | 0 | 0 | 3 | - | - | - |
| Tray / system (T-18.x) | 0 | 0 | 2 | 2 | - | - | - |
| **Total** | **42** | **8** | **17** | **67** | - | - | - |

## 3. Performance Baselines

### 3.1 Latency

| Metric | Target | Chinese Balanced | Multilingual Offline | Apple Native |
|---|---|---|---|---|
| `cold_start_latency_ms` | SV<2000 / FN<8000 / Apple<1500 | PENDING | PENDING | PENDING |
| `steady_p50_latency_ms` | SV<300 / FN<2000 / Apple<800 | PENDING | PENDING | PENDING |
| `steady_p95_latency_ms` | <2× P50 | PENDING | PENDING | PENDING |
| `model_load_ms` | SV<1500 / FN<5000 / Apple<200 | PENDING | PENDING | PENDING |
| `apple_speech_first_partial_ms` | <600 | n/a | n/a | PENDING |

| CT-Punc metric | Target | Measured |
|---|---|---|
| `punc_zh_first_call_ms` | <500 | PENDING |
| `punc_zh_steady_ms` | <30 | PENDING |

| Engine swap | Target | Measured |
|---|---|---|
| `engine_swap_ms` (SV→FN) | <3000 | PENDING |
| `engine_swap_ms` (FN→SV) | <3000 | PENDING |

### 3.2 Accuracy

| Metric | Target | zh_short | zh_long | mixed | en |
|---|---|---|---|---|---|
| WER (char-level) | zh<5% / en<8% / mixed<15% | PENDING | PENDING | PENDING | PENDING |
| `punctuation_density` | zh 0.04-0.08 | PENDING | PENDING | PENDING | n/a |
| `hotword_recall` | ≥80% | PENDING | PENDING | PENDING | PENDING |
| `tag_strip_rate` (SenseVoice meta) | =0 | PENDING | PENDING | n/a | n/a |
| `itn_pass_rate` | 31/31 | PENDING | n/a | n/a | n/a |

### 3.3 Resource

| Preset | RSS peak | CPU% (idle) | Model disk | Memory drift over 30s idle |
|---|---|---|---|---|
| Chinese Balanced | PENDING | PENDING | 152 MB | PENDING |
| Multilingual Offline | PENDING | PENDING | 1.0 GB | PENDING |
| Apple Native | PENDING | PENDING | 0 MB | PENDING |

### 3.4 Stability

| Test | Target | Result |
|---|---|---|
| 100 consecutive transcriptions, no crash | 100% | PENDING |
| Engine panic recovery (HANDY_FORCE_TRANSCRIPTION_FAILURE) | auto-reload | PENDING |
| Apple Speech 30s timeout no hang (T-13.4) | timer fires | PENDING |
| Mutex non-poisoning after panic | next call OK | PENDING |

### 3.5 UX

| Metric | Target | Measured |
|---|---|---|
| Settings UI preset switch response | <200ms | PENDING |
| Overlay render FPS (long partial stream) | ≥30 | PENDING |
| Tray feedback latency | <100ms | PENDING |

## 4. Detailed Test Cases

### 4.1 Pass List

PENDING — populated by Wave 1-3 runners.

### 4.2 Fail / Regression

PENDING — each entry must include reproduction steps + log excerpt + screenshot path.

### 4.3 Known Issues / Skipped

PENDING.

## 5. Comparison with v0.8.6

### 5.1 Performance delta

| Metric | v0.8.6 | v0.8.7 | Δ |
|---|---|---|---|
| chinese_balanced steady_p50 | 925 ms (Tauri pipeline) | PENDING | PENDING |
| multilingual_offline steady_p50 | 3510 ms | PENDING | PENDING |
| apple_native steady_p50 | hung | PENDING | PENDING |
| chinese_balanced punc density | 0.064 | PENDING | PENDING |
| multilingual_offline punc density | 0.085 | PENDING | PENDING |
| apple_native punc density | 0.008 (CT-Punc skipped) | PENDING (CJK fallback fixed) | PENDING |

### 5.2 Fix verification

| v0.8.7 fix | Verification | Result |
|---|---|---|
| CT-Punc CJK fallback (apple_native auto language) | T-2.3 | PENDING |
| Apple Speech permission hang (5s auth timeout + GCD timer) | T-13.x | PENDING |
| Apple Speech error classification (PERM_DENIED / TIMEOUT / ENGINE) | T-13.5 | PENDING |
| `get_speech_recognition_permission` command | T-13.1/2/3 | PENDING |
| Bench latency inflation (per-item spawn_blocking) | T-14.2 | PENDING |
| Cold-start vs steady-state breakdown in JSON | T-14.1 | PENDING |
| Punc model download UI (state machine + progress) | T-12.1 | PENDING |
| Startup legacy-dir cleanup + missing-toast | T-12.2 | PENDING |
| Logo refresh (mic→waveform) | visual | PENDING |

### 5.3 New regressions

PENDING.

## 6. Recommendations

### 6.1 Wave 4 optimization candidates (ROI-ranked)

PENDING — populated after Wave 1-3 data identifies bottlenecks.

| Optimization | Hypothesis | Expected gain | Risk |
|---|---|---|---|
| O-1 SenseVoice steady_p50 → <300 ms | apply_custom_words O(N×M) fuzzy | -300 ms | may miss some fuzzy matches |
| O-2 FunASR-Nano cold_start → <3 s | OnnxRuntime JIT cache to disk | -5 s first-launch saved on relaunch | first-time still slow |
| O-3 Apple Speech under Tauri runtime | scheduling mode | unblocks bench | partial ordering risk |
| O-4 punc_zh OnceCell global | already used; verify correctness | -100 ms first call | +50 MB RSS |
| O-5 Profile hot-swap < 500 ms | reuse ORT session for same family | -1 s | cross-family inapplicable |

### 6.2 v0.8.8 follow-ups

PENDING.

## Appendix

### A. Raw benchmark JSONs

`benchmark/results/v0.8.7/` — generated by `--bench-mode {asr,punc-only,chain,swap}` runs.

### B. Environment

| Field | Value |
|---|---|
| OS | macOS PENDING |
| Hardware | Apple Silicon PENDING |
| sherpa-onnx version | 1.13.x (see Cargo.lock) |
| transcribe-rs version | (see Cargo.lock) |
| CT-Punc model | `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8` (76 MB) |
| Test harness commit | PENDING (`git rev-parse HEAD` at run time) |

### C. Test harness

- Bench binary: `src-tauri/src/commands/benchmark.rs` + `cli.rs --bench-mode`
- Python eval: `benchmark/eval.py` (single + `--compare` modes)
- Dataset manifest: `benchmark/dataset/dataset_manifest.json` (5 subsets)
- Diary integration tests: `src-tauri/src/actions.rs::diary_tests`
- Apple Speech permission script: `scripts/test_apple_perm_states.sh`
- Resource monitor: `scripts/measure_resource.sh`

### D. Test plan source

Plan agent output saved at task #23 (handoff). 67 cases, 4 waves, ~3-4 dev-day estimate.
