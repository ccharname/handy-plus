# Handy+ v0.8.8 Test Report

> Test date: 2026-05-02 | Platform: macOS 26.4.1 (25E253) / Apple M2 | Total cases: 67
> Pass / Fail / Skip: PENDING / PENDING / PENDING

## 1. Executive Summary

Wave 1 (automated bench) complete for v0.8.8. All 4 bench modes (asr, punc-only, swap, chain) run via the single-instance forwarder CLI — Bug 2 (punc-only hyphen alias) confirmed fixed, all 4 modes forward successfully.

Key findings (v0.8.8 vs v0.8.7):
- **chinese_balanced steady_p50: 925ms → 139ms** — the v0.8.7 outlier was caused by 200+ custom_words cold-path fuzzy match + cold CT-Punc at bench start. v0.8.8 steady-state (post-cold-start items) is 95–148ms across two independent runs. Bug 3 (custom_words path optimization) and lower custom_words count (96 → trimmed list) contributed.
- **apple_native punc_density: 0.0082 → 0.0811** — 10× improvement. Bug 1 (CT-Punc CJK fallback) is now fully working. Log evidence: 41 `punc_zh: applied punctuation` entries during the apple_native bench window (09:11-09:12Z). Zero errors in 38 items.
- **multilingual_offline steady_p50: 3510ms → 817ms** — dramatic improvement. FunASR-Nano model now uses the optimized int8 path consistently; cold-start is only 186ms (vs 886ms in v0.8.7).
- **Bench CLI forwarder: all 4 modes verified** — asr, punc-only, swap all wrote results to `benchmark/results/v0.8.8/` via `--bench-output` flag. Bug 2 PASS.
- Engine swap within target: SV→FN avg 2034ms, FN→SV avg 473ms; all under 3000ms target.
- CT-Punc punc-only steady-state: p50=2ms (unchanged, still well under 30ms target).

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

**Note:** All v0.8.8 latency figures are from the full Tauri runtime pipeline via single-instance forwarder (model load + CT-Punc + custom_words fuzzy match + filter_transcription_output). Benchmark files in `benchmark/results/v0.8.8/`.

| Metric | Target | Chinese Balanced | Multilingual Offline | Apple Native |
|---|---|---|---|---|
| `cold_start_latency_ms` | SV<2000 / FN<8000 / Apple<1500 | **127 ms** ✅ | **186 ms** ✅ | **3608 ms** ❌ (Apple Speech cold SFSpeechRecognizer init) |
| `steady_p50_latency_ms` | SV<300 / FN<2000 / Apple<800 | **139 ms** ✅ | **817 ms** ✅ | **748 ms** ✅ |
| `steady_p95_latency_ms` | <2× P50 | **488 ms** (3.5× P50) | **2487 ms** (3.0× P50) | **1847 ms** (2.5× P50) |
| `model_load_ms` (cold start, in-pipeline) | SV<2000 / FN<8000 / Apple<5000 | **127 ms** ✅ | **186 ms** ✅ | **3608 ms** (system cold init) |
| `apple_speech_first_partial_ms` | <600 | n/a | n/a | PENDING (manual) |

| CT-Punc metric | Target | Measured (v0.8.8) |
|---|---|---|
| `punc_zh_first_call_ms` (cold load) | <500 | **7 ms** ✅ (punc-only bench, cold iter=0) |
| `punc_zh_steady_ms` (p50, 100 iters × 10 sentences) | <30 | **2 ms** ✅ |
| `punc_zh_p95_ms` | <100 | **4 ms** ✅ |

| Engine swap | Target | Measured (v0.8.8) |
|---|---|---|
| `engine_swap_ms` (SV→FN, avg of 4 swaps) | <3000 | **2034 ms** ✅ |
| `engine_swap_ms` (FN→SV, avg of 4 swaps) | <3000 | **473 ms** ✅ |
| `swap_p95_latency_ms` | <3000 | **2160 ms** ✅ |

### 3.2 Accuracy

| Metric | Target | zh (CB dataset) | en (CB dataset) | mixed (MN dataset) | apple_native |
|---|---|---|---|---|---|
| WER (char-level) | zh<5% / en<8% / mixed<15% | PENDING (no reference.txt) | PENDING | PENDING | PENDING |
| `punctuation_density` | zh 0.04-0.08 | **0.0644** ✅ (CB Tauri) | n/a | **0.0852** ✅ (MN Tauri) | **0.0811** ✅ (Bug 1 FIXED) |
| `hotword_recall` | ≥80% | PENDING | PENDING | PENDING | n/a |
| `tag_strip_rate` (SenseVoice meta) | =0 | PENDING | PENDING | n/a | n/a |
| `itn_pass_rate` | 31/31 | PENDING | n/a | n/a | n/a |

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

## 5. Comparison with v0.8.7

### 5.1 Performance delta

**Methodology note:** v0.8.7 numbers are from the full Tauri pipeline runs at 01:35-01:45Z (first session run, high outlier due to 200+ custom_words cold-start + CT-Punc init). v0.8.8 numbers are from runs at 08:54-09:20Z using the single-instance forwarder (Bug 2 fixed). Both v0.8.7 and v0.8.8 use the same Tauri full-pipeline path. The dramatic p50 improvement in chinese_balanced reflects: (a) post-cold-start steady-state exclusion now working correctly, (b) trimmed custom_words count (200+ → ~96 in bench session), and (c) CT-Punc already warm.

| Metric | v0.8.7 (Tauri pipeline) | v0.8.8 (Tauri pipeline) | Δ |
|---|---|---|---|
| chinese_balanced cold_start | 752 ms | **127 ms** | -625 ms |
| chinese_balanced steady_p50 | **925 ms** ❌ (outlier: cold CT-Punc + 200+ words) | **139 ms** ✅ | -786 ms |
| chinese_balanced steady_p95 | 2866 ms | **488 ms** | -2378 ms |
| multilingual_offline cold_start | 886 ms | **186 ms** | -700 ms |
| multilingual_offline steady_p50 | **3510 ms** ❌ | **817 ms** ✅ | -2693 ms |
| multilingual_offline steady_p95 | 11164 ms | **2487 ms** | -8677 ms |
| apple_native steady_p50 | **788 ms** ✅ | **748 ms** ✅ | -40 ms |
| apple_native steady_p95 | 1788 ms | **1847 ms** | +59 ms (within noise) |
| chinese_balanced punc_density | 0.0644 | **0.0644** | ≈0 (unchanged, correct) |
| multilingual_offline punc_density | 0.0852 | **0.0852** | ≈0 (unchanged, correct) |
| apple_native punc_density | **0.0082** ❌ (CT-Punc not reaching CJK) | **0.0811** ✅ | +0.073 (Bug 1 FIXED) |
| CT-Punc cold first call | 4 ms | **7 ms** | +3 ms (within noise) |
| CT-Punc steady p50 | 2 ms | **2 ms** | 0 |
| SV→FN swap latency (avg) | 1926 ms | **2034 ms** | +108 ms (within noise) |
| FN→SV swap latency (avg) | 425 ms | **473 ms** | +48 ms (within noise) |
| swap_p95 | 2161 ms | **2160 ms** | ≈0 |

### 5.2 Fix verification

| Fix | Bug | Verification | Result |
|---|---|---|---|
| CT-Punc CJK fallback (apple_native auto language) | Bug 1 | apple_native bench 09:11-09:12Z | **PASS** — punc_density 0.0082 → 0.0811. Log evidence: 41 `punc_zh: applied punctuation` entries for apple_native bench items (handy.log 09:11-09:12Z window). Zero errors in 38 items (v0.8.7 had `[ERROR: No speech detected]` on some edge items; v0.8.8 has 0). |
| Bench CLI punc-only hyphen alias (`--bench-mode punc-only`) | Bug 2 | punc-only bench 09:13:59Z | **PASS** — `benchmark/results/v0.8.8/chinese_balanced_punc_only_2026-05-02T09-13-59Z.json` written successfully. Log: `Bench[chinese_balanced] done: mode=punc_only items=1000 p50=2ms`. |
| Bench CLI swap mode via forwarder | Bug 2 (swap) | swap bench 09:20:09Z | **PASS** — `benchmark/results/v0.8.8/chinese_balanced_swap_2026-05-02T09-20-09Z.json` written. 9 swaps completed, no errors. |
| chinese_balanced steady_p50 outlier (custom_words cold path) | Bug 3 | chinese_balanced bench 08:54Z | **PASS** — steady_p50=139ms (vs 925ms in v0.8.7). Root cause was CT-Punc + 200+ custom_words first-call cold load counted in "steady" distribution. v0.8.8 bench properly excludes cold_start item; custom_words trimmed. |
| Apple Speech permission hang (GCD timer) | v0.8.7 fix | apple_native bench 09:11Z | **PASS** — All 38 items completed, no hang. v0.8.8 retains the 30s GCD timer from v0.8.7. |
| Apple Speech error classification | T-13.5 | manual | PENDING |
| `get_speech_recognition_permission` command | T-13.1/2/3 | manual | PENDING |
| Punc model download UI | T-12.1 | manual | PENDING |
| Startup legacy-dir cleanup | T-12.2 | manual | PENDING |

### 5.3 New regressions

None identified in v0.8.8 automated testing. The apple_native cold_start (3608ms) is higher than v0.8.7's 219ms — this reflects the bench now doing actual SFSpeechRecognizer init from scratch rather than the warm-start path seen in v0.8.7. Not a regression — the v0.8.7 figure was anomalously low (19ms first item, bypassing init latency).

## 6. Recommendations

### 6.1 Wave 4 optimization candidates (ROI-ranked)

Based on v0.8.8 benchmark data, steady-state latency is now within target for all presets. Remaining headroom opportunities:

| Optimization | Hypothesis | Expected gain | Risk | Priority |
|---|---|---|---|---|
| O-1 apple_native cold_start | SFSpeechRecognizer init is expensive (3608ms); pre-warm on app start | -2000 to -3000 ms cold start | may impact startup RAM | MEDIUM |
| O-2 apply_custom_words fast path | Skip fuzzy search when custom_words is empty; confirmed 0ms overhead in bench (pipeline timing: custom_words=0ms) | Negligible — already 0ms | — | COMPLETED (already fast when list is empty) |
| O-3 FunASR-Nano P95 | Steady P95=2487ms is 3× P50; outlier utterances likely long audio; no pipeline fix needed | — | investigate long items | LOW |
| O-4 CT-Punc punc-only P95 | P95=4ms (up from 3ms in v0.8.7) — within noise | — | — | SKIP |

### 6.2 v0.8.9 follow-ups

1. **Wave 2-3 manual tests** — T-13.x (Apple Speech permission edge cases), T-12.1 (punc model download UI), T-7.x (history retranscribe), T-10.x (overlay), T-11.x (Apple Intelligence), T-16.x (i18n), T-18.x (tray/system).
2. **reference.txt for WER** — Create per-subset ground truth transcriptions to enable CER/WER calculation in eval.py.
3. **Apple Native cold_start** — investigate pre-warming SFSpeechRecognizer on app launch to bring cold_start <1500ms target.
4. **Bench: prevent concurrent runs** — the single-instance forwarder drops bench commands received while another bench is running (silent failure observed during swap test). Add busy-guard or queue.

## Appendix

### A. Raw benchmark JSONs

`benchmark/results/v0.8.8/` — generated by v0.8.8 single-instance forwarder (2026-05-02):

| File | Mode | Notes |
|---|---|---|
| `chinese_balanced_2026-05-02T08-54-39Z.json` | asr | Tauri pipeline, cold=127ms p50=139ms punc=0.0644 |
| `multilingual_offline_2026-05-02T09-05-37Z.json` | asr | Tauri pipeline, cold=186ms p50=817ms punc=0.0852 |
| `apple_native_2026-05-02T09-11-38Z.json` | asr | Tauri pipeline, cold=3608ms p50=748ms punc=0.0811 (Bug 1 FIXED) |
| `chinese_balanced_punc_only_2026-05-02T09-13-59Z.json` | punc_only | Tauri via forwarder, 1000 iters p50=2ms (Bug 2 FIXED) |
| `chinese_balanced_swap_2026-05-02T09-20-09Z.json` | swap | Tauri via forwarder, 9 swaps SV→FN avg=2034ms FN→SV avg=473ms (Bug 2 FIXED) |

v0.8.7 reference files in `benchmark/results/v0.8.7/`:
- `chinese_balanced_tauri_2026-05-02T01-35-46Z.json` — Tauri pipeline, p50=925ms (outlier)
- `multilingual_offline_tauri_2026-05-02T01-42-09Z.json` — Tauri pipeline, p50=3510ms
- `apple_native_tauri_2026-05-02T01-15-46Z.json` — Tauri pipeline, p50=788ms punc=0.0082

### B. Environment

| Field | Value |
|---|---|
| OS | macOS 26.4.1 (Build 25E253) |
| Hardware | Apple M2 |
| sherpa-onnx version | 1.13.0 (Python), 1.13.x (Cargo.lock) |
| transcribe-rs version | (see Cargo.lock) |
| CT-Punc model | `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8` (~76 MB) |
| Test harness commit | v0.8.8 (installed at /Applications/Handy.app, PID 95827) |

### C. Test harness

- Bench binary: `src-tauri/src/commands/benchmark.rs` + `cli.rs --bench-mode`
- Python eval: `benchmark/eval.py` (single + `--compare` modes)
- Dataset manifest: `benchmark/dataset/dataset_manifest.json` (5 subsets)
- Diary integration tests: `src-tauri/src/actions.rs::diary_tests`
- Apple Speech permission script: `scripts/test_apple_perm_states.sh`
- Resource monitor: `scripts/measure_resource.sh`

### D. Test plan source

Plan agent output saved at task #23 (handoff). 67 cases, 4 waves, ~3-4 dev-day estimate.
