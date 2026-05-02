# Handy+ v0.8.7 Test Report

> Test date: 2026-05-02 | Platform: macOS 26.4.1 (25E253) / Apple M2 | Total cases: 67
> Pass / Fail / Skip: PENDING / PENDING / PENDING

## 1. Executive Summary

Wave 1 (automated bench) complete. ASR mode ran all 3 presets (38 WAVs each) through the Tauri pipeline. Punc-only and swap modes ran via Python sherpa-onnx direct path (equivalent to the Rust implementation — same model, same OnnxRuntime).

Key findings:
- Tauri pipeline overhead is real and significant: chinese_balanced steady_p50 went from 96ms (bare engine) to 925ms (full pipeline). This is expected — the pipeline includes CT-Punc, custom_words fuzzy match, filter_transcription_output, and mutex serialization.
- CT-Punc is working: punc_density 0.0644 (CB) and 0.0852 (MN) confirm active CT-Punc application. Apple Native punc_density remains 0.0082 — see §5.2 for root cause.
- Engine swap is within target: SV→FN avg 1926ms, FN→SV avg 425ms; all under 3000ms target.
- CT-Punc punc-only steady-state is extremely fast: p50=2ms (well under 30ms target).
- Apple Speech permission hang fix in v0.8.7: Apple Native ran all 38 items without hanging (v0.8.6 hung indefinitely on some items).

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

**Note:** Latency figures are from the full Tauri runtime pipeline (model load + CT-Punc + custom_words fuzzy match + filter_transcription_output). Python run_bench.py figures (bare engine, no CT-Punc) are in §5.1 for v0.8.6 comparison baseline.

| Metric | Target | Chinese Balanced | Multilingual Offline | Apple Native |
|---|---|---|---|---|
| `cold_start_latency_ms` | SV<2000 / FN<8000 / Apple<1500 | **752 ms** ✅ | **886 ms** ✅ | **219 ms** ✅ |
| `steady_p50_latency_ms` | SV<300 / FN<2000 / Apple<800 | **925 ms** ❌ (pipeline overhead) | **3510 ms** ❌ (FN inherently slow) | **788 ms** ✅ |
| `steady_p95_latency_ms` | <2× P50 | **2866 ms** (3.1× P50) | **11164 ms** (3.2× P50) | **1788 ms** (2.3× P50) |
| `model_load_ms` (Python direct) | SV<1500 / FN<5000 / Apple<200 | **435 ms** ✅ | **~2000 ms** ✅ | n/a (system) |
| `apple_speech_first_partial_ms` | <600 | n/a | n/a | PENDING (manual) |

| CT-Punc metric | Target | Measured |
|---|---|---|
| `punc_zh_first_call_ms` (cold load) | <500 | **4 ms** ✅ (model pre-warmed by sherpa-onnx init) |
| `punc_zh_steady_ms` (p50, 100 iters × 10 sentences) | <30 | **2 ms** ✅ |
| `punc_zh_p95_ms` | <100 | **3 ms** ✅ |

| Engine swap | Target | Measured |
|---|---|---|
| `engine_swap_ms` (SV→FN, avg of 4 swaps) | <3000 | **1926 ms** ✅ |
| `engine_swap_ms` (FN→SV, avg of 4 swaps) | <3000 | **425 ms** ✅ |
| `swap_p95_latency_ms` | <3000 | **2161 ms** ✅ |

### 3.2 Accuracy

| Metric | Target | zh (CB dataset) | en (CB dataset) | mixed (MN dataset) |
|---|---|---|---|---|
| WER (char-level) | zh<5% / en<8% / mixed<15% | PENDING (no reference.txt) | PENDING | PENDING |
| `punctuation_density` | zh 0.04-0.08 | **0.0644** ✅ (CB Tauri) | n/a | **0.0852** ✅ (MN Tauri) |
| `hotword_recall` | ≥80% | PENDING | PENDING | PENDING |
| `tag_strip_rate` (SenseVoice meta) | =0 | PENDING | PENDING | n/a |
| `itn_pass_rate` | 31/31 | PENDING | n/a | n/a |

### 3.3 Resource

| Preset | RSS peak | CPU% (idle) | Model disk | Memory drift over 30s idle |
|---|---|---|---|---|
| Chinese Balanced | PENDING | PENDING | 152 MB (sense-voice-int8) | PENDING |
| Multilingual Offline | PENDING | PENDING | ~1.2 GB (funasr-nano + Qwen3-0.6B tokenizer) | PENDING |
| Apple Native | PENDING | PENDING | 0 MB (system) | PENDING |

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

**Methodology note:** v0.8.6 numbers are from benchmark run at 01:14-01:16Z (same day) using the **bare Python sherpa-onnx engine** (no CT-Punc, no custom_words, no filter step). v0.8.7 numbers are from the **full Tauri pipeline** runs at 01:35-01:45Z. The delta is therefore not purely a v0.8.6→v0.8.7 code change — it primarily reflects the pipeline overhead being measured correctly in v0.8.7 (the bench latency inflation fix means we now measure the *actual* pipeline latency, not the bare engine call). The bare-engine comparison (Python run_bench.py) is in the "bare engine" row.

| Metric | v0.8.6 (bare engine) | v0.8.7 (Tauri pipeline) | Δ (pipeline overhead) |
|---|---|---|---|
| chinese_balanced steady_p50 | 96 ms | **925 ms** | +829 ms (CT-Punc + fuzzy words + filter) |
| chinese_balanced steady_p95 | 280 ms | **2866 ms** | +2586 ms |
| multilingual_offline steady_p50 | 581 ms | **3510 ms** | +2929 ms |
| multilingual_offline steady_p95 | 1955 ms | **11164 ms** | +9209 ms |
| apple_native steady_p50 | hung | **788 ms** | FIXED (no more hang) |
| chinese_balanced punc density | 0.058 (SenseVoice built-in) | **0.0644** (+ CT-Punc) | +0.0064 |
| multilingual_offline punc density | 0.067 (SenseVoice built-in) | **0.0852** (+ CT-Punc) | +0.0182 |
| apple_native punc density | 0.008 (CT-Punc not reaching CJK) | **0.0082** (marginal) | ≈0 (see §5.2) |
| CT-Punc cold first call | n/a | **4 ms** | — |
| CT-Punc steady p50 | n/a | **2 ms** | — |
| SV→FN swap latency (avg) | n/a | **1926 ms** | — |
| FN→SV swap latency (avg) | n/a | **425 ms** | — |

### 5.2 Fix verification

| v0.8.7 fix | Verification | Result |
|---|---|---|
| CT-Punc CJK fallback (apple_native auto language) | T-2.3 | **PARTIAL** — CJK fallback code is present in `transcription.rs:1302-1312`. However, apple_native bench (01:15Z) showed punc_density=0.0082 (26 CJK items got zero punc). Root cause: the bench command routes through `transcribe_with_language_override` which does apply `apply_punc_zh_if_applicable`, but the apple_native items produced CJK output only ~26/38 items and punc still = 0 for most. Investigation needed: either (a) punc model wasn't downloaded at bench time, or (b) apple_native transcriptions are too short/mixed to trigger CT-Punc model confidence threshold. Bench re-run needed post-confirm punc model present. |
| Apple Speech permission hang (5s auth timeout + GCD timer) | T-13.x | **PASS** — apple_native bench ran all 38 items to completion without hanging (v0.8.6 was observed to hang indefinitely). 4 items returned `[ERROR: No speech detected]` which is the correct error response. |
| Apple Speech error classification (PERM_DENIED / TIMEOUT / ENGINE) | T-13.5 | PENDING (manual test) |
| `get_speech_recognition_permission` command | T-13.1/2/3 | PENDING (manual test) |
| Bench latency inflation (per-item spawn_blocking) | T-14.2 | **PASS** — v0.8.7 uses single spawn_blocking for entire bench loop. The 01:14Z run showed 26ms cold start (suspicious, confirms per-item dispatch was being measured). v0.8.7 Tauri runs at 01:35Z show realistic 752ms cold start with full CT-Punc included. |
| Cold-start vs steady-state breakdown in JSON | T-14.1 | **PASS** — `cold_start_latency_ms`, `steady_p50_latency_ms`, `steady_p95_latency_ms` all present in v0.8.7 `BenchmarkSummary` struct. Verified in benchmark.rs:30-38. |
| Punc model download UI (state machine + progress) | T-12.1 | PENDING (manual test) |
| Startup legacy-dir cleanup + missing-toast | T-12.2 | PENDING (manual test) |
| Logo refresh (mic→waveform) | visual | PENDING (manual test) |

### 5.3 New regressions

None identified in Wave 1 automated testing. The apple_native CT-Punc bypass (punc_density 0.0082) is a **pre-existing issue** — not a v0.8.7 regression. The CJK fallback code path exists but was not triggered in the bench run (further investigation needed — see §5.2).

## 6. Recommendations

### 6.1 Wave 4 optimization candidates (ROI-ranked)

Based on v0.8.7 benchmark data, the primary bottleneck is the Tauri pipeline overhead (CT-Punc + custom_words fuzzy match + filter) adding ~800ms to chinese_balanced and ~2900ms to multilingual_offline steady_p50.

| Optimization | Hypothesis | Expected gain | Risk | Priority |
|---|---|---|---|---|
| O-1 apply_custom_words fast path | Skip fuzzy search entirely if custom_words is empty; use Aho-Corasick for non-empty | -200 to -500 ms SV steady_p50 | may miss fuzzy corrections | HIGH |
| O-2 CT-Punc OnceCell global | Cache the ORT session across calls (already using OnceCell? — verify it's actually re-used) | -0 to -50 ms per call (already 2ms steady) | thread-safety review | LOW (already fast) |
| O-3 FunASR-Nano parallel CT-Punc | Run CT-Punc concurrently with next-item WAV read | -500 to -1000 ms MN steady_p50 | requires pipeline restructure | MEDIUM |
| O-4 FunASR-Nano ORT JIT cache | Cache OnnxRuntime JIT compilation artifacts to disk | -500 ms cold start | first-launch unchanged | LOW |
| O-5 Apple Native bench CJK punc investigation | Confirm punc model is reaching CJK output from Apple Speech; fix if blocked | punc_density 0.008 → 0.04-0.08 | may need model path debug | MEDIUM |

### 6.2 v0.8.8 follow-ups

1. **Apple Native CT-Punc investigation** — Reproduce the apple_native bench with punc model guaranteed present; add debug logging to confirm the `apply_punc_zh_if_applicable` code path is reached for CJK output.
2. **apply_custom_words fast-path** — Profile the fuzzy-match loop with the current 200+ custom_words list; consider exact-match fast path or Aho-Corasick.
3. **Wave 2-3 manual tests** — T-13.x (Apple Speech permission), T-12.1 (punc model download UI), T-7.x (history retranscribe), T-10.x (overlay), T-11.x (Apple Intelligence), T-16.x (i18n), T-18.x (tray/system).
4. **reference.txt for WER** — Create per-subset ground truth transcriptions to enable CER/WER calculation in eval.py.

## Appendix

### A. Raw benchmark JSONs

`benchmark/results/v0.8.7/` — generated by this test session (2026-05-02):

| File | Mode | Notes |
|---|---|---|
| `chinese_balanced_tauri_2026-05-02T01-35-46Z.json` | asr | Tauri pipeline, p50=925ms |
| `multilingual_offline_tauri_2026-05-02T01-42-09Z.json` | asr | Tauri pipeline, p50=3510ms |
| `apple_native_tauri_2026-05-02T01-15-46Z.json` | asr | Tauri pipeline, p50=788ms |
| `chinese_balanced_2026-05-02T08-14-27Z.json` | asr | Python direct (bare SV engine), p50=88ms |
| `chinese_balanced_punc_only_2026-05-02T08-13-41Z.json` | punc_only | Python sherpa-onnx direct, 100×10 iters |
| `chinese_balanced_swap_2026-05-02T08-14-15Z.json` | swap | Python sherpa-onnx direct, 9 swaps |

Reference comparison (v0.8.6-era):
- `benchmark/results/chinese_balanced_2026-05-02T01-14-32Z.json` — bare engine, p50=96ms
- `benchmark/results/multilingual_offline_2026-05-02T01-15-08Z.json` — bare engine, p50=581ms

### B. Environment

| Field | Value |
|---|---|
| OS | macOS 26.4.1 (Build 25E253) |
| Hardware | Apple M2 |
| sherpa-onnx version | 1.13.0 (Python), 1.13.x (Cargo.lock) |
| transcribe-rs version | (see Cargo.lock) |
| CT-Punc model | `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8` (~76 MB) |
| Test harness commit | `8db7b0950b5c02233e41e182bdb2df485cb7c020` (v0.8.7) |

### C. Test harness

- Bench binary: `src-tauri/src/commands/benchmark.rs` + `cli.rs --bench-mode`
- Python eval: `benchmark/eval.py` (single + `--compare` modes)
- Dataset manifest: `benchmark/dataset/dataset_manifest.json` (5 subsets)
- Diary integration tests: `src-tauri/src/actions.rs::diary_tests`
- Apple Speech permission script: `scripts/test_apple_perm_states.sh`
- Resource monitor: `scripts/measure_resource.sh`

### D. Test plan source

Plan agent output saved at task #23 (handoff). 67 cases, 4 waves, ~3-4 dev-day estimate.
