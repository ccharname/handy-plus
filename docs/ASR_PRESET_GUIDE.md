# Handy+ ASR Preset Guide

Handy+ ships three pre-tuned ASR (automatic speech recognition) configurations, exposed in **Settings → Models → ASR Presets**. Each preset is a one-click bundle of {engine, language, punctuation layer, hot-words wiring} that lets you switch the speech stack without poking individual settings.

This document explains what each preset is good for, the trade-offs between them, and how to evaluate them on your own audio using `benchmark/`.

---

## TL;DR — Picking a preset

| Use case | Preset | Why |
|---|---|---|
| Mostly Chinese dictation (writing, chat, notes) | **Chinese Balanced** | SenseVoice is fast (≈70 ms / 10 s clip), CT-Transformer-Punc adds standard Chinese punctuation, low memory footprint (~152 MB) |
| Multilingual mixed input (zh+en+ja+...) — must run offline | **Multilingual Offline** | FunASR-Nano LLM-decoder handles 8+ languages well; CT-Punc still patches Chinese punctuation |
| Fast English short commands, latency-critical, system-language-driven | **Apple Native** | SFSpeechRecognizer first-token latency is the lowest; v0.8.6 CT-Punc kicks in whenever recognised text contains CJK (content-based fallback — no longer depends on system locale) |
| Anything else / power user | **Advanced (Customized)** — keep the per-field controls below the preset cards |

If you don't know which one to start with: **Chinese Balanced is the v0.8.3-handy-plus.6 default**. Apply it once and start dictating.

### Benchmark numbers (v0.8.6, M-series, 38 clips, 331 s audio)

> **How v0.8.6 benchmark numbers are produced**  
> Numbers in this table come from `run_asr_benchmark` — a Tauri command that runs inside the full
> Tauri async runtime with `spawn_blocking` dispatch.  This is a realistic end-to-end measurement
> (it includes `transcribe_with_language_override` pipeline overhead: `get_settings`, language
> validation, custom-word correction, filler-word filter, CT-Punc layer) but it also means:  
> - **Cold-start item** (first WAV): includes OnnxRuntime JIT compilation, CT-Punc `OnceCell` init,
>   and any one-time setup costs.  Expect 3–8× higher than steady-state.  
> - **Steady-state items** (2nd WAV onward): model is already warm in the engine mutex; these
>   numbers are close to real interactive latency.  
> - v0.8.6 fix: the loop now runs inside a **single** `spawn_blocking` call so per-item Tokio
>   dispatch overhead (~5–50 ms) no longer inflates measurements; `steady_p50` is the reliable
>   production number.  
>
> (38 WAVs: 5 user recordings + 5 SenseVoice clips + 24 FunASR-Nano clips + 4 edge-case clips).  
> ⚠️ Apple Native latency numbers are from v0.8.5 run_bench.py (Tauri runtime bench hangs on
> SFSpeechRecognizer first-use with on-device model — bug tracked, fix pending).

| Metric | Chinese Balanced | Multilingual Offline | Apple Native |
|---|---|---|---|
| Engine | SenseVoice-int8 | FunASR-Nano | Apple Speech |
| P50 latency (ms) — all items | **925** | 3510 | 788 ¹ |
| P95 latency (ms) — all items | **2866** | 11164 | 1788 ¹ |
| Cold-start (1st item, ms) | (pending re-run) | (pending re-run) | n/a |
| Steady-state P50 (ms) | (pending re-run) | (pending re-run) | n/a |
| Steady-state P95 (ms) | (pending re-run) | (pending re-run) | n/a |
| Punctuation density | 0.0644 | 0.0852 | ~0.04–0.06 ² |
| Errors (38 clips) | 0 | 0 | 4 (silence/noise) |

¹ Apple Native latency from v0.8.5 run_bench.py (Tauri CLI bench pending fix).  
² v0.8.6 CJK fallback means CT-Punc now fires on Chinese text from Apple Speech (was 0.0082 in
v0.8.5 when CT-Punc was skipped for `auto`+non-zh app_language). Exact density pending runtime bench.

> **Why v0.8.5 direct-run numbers were faster**  
> v0.8.5 benchmarking used `run_bench.py` which called the engine Python bindings directly, bypassing
> the Tauri pipeline. That measured raw engine inference only (~96 ms for SenseVoice on a 10 s clip).
> The v0.8.6 Tauri runtime numbers (P50 = 925 ms) include the full pipeline. The gap narrowed
> significantly with the single-`spawn_blocking` fix: the old per-item dispatch added ~50–200 ms per
> item in async wake + mutex contention overhead.  
>
> **Practical implication for interactive use:** the user presses the shortcut, speaks, releases —
> the transcription pipeline fires once. This maps to the *steady-state* path (model already loaded).
> The cold-start penalty (first transcription after launch or after model unload) is a one-time cost.

Key takeaways (v0.8.6):
- **Chinese Balanced is 3.8× faster at P50** than Multilingual Offline on the real Tauri pipeline.
- **CT-Punc is active on all three presets** — the CJK content-based fallback (v0.8.6 fix) ensures
  Apple Native / Multilingual Offline auto-mode transcriptions get punctuated whenever CJK chars appear.
- **Apple Native gracefully errors** on silence/noise clips (`No speech detected`) — clean degradation.
- **FunASR-Nano** has the highest punc density (0.0852) confirming strong sentence segmentation on
  multi-lingual content; latency cost is 3.8× vs SenseVoice at P50.

---

## What each preset actually does

### 1. Chinese Balanced (`chinese_balanced`)

| Layer | Setting |
|---|---|
| Engine | SenseVoice-int8 (transcribe-rs path) |
| Language | `zh-Hans` |
| Punctuation | CT-Transformer-Punc enabled (`punc_zh_enabled = true`) |
| Hot-words boost | Default 2.0 (used only when sherpa-onnx path is selected) |
| ITN (number normalization) | Disabled in engine; handled downstream by `itn_zh.rs` |

**Benchmark results** (v0.8.6, 2026-05-02, 38 clips, 331 s audio, M-series, Tauri runtime):
P50 = **925 ms** | P95 = **2866 ms** | punc density = 0.0644 | errors = 0  
Steady-state P50/P95 pending re-run with v0.8.6 single-spawn_blocking fix.

**Strengths**
- Fastest end-to-end for Chinese: SenseVoice-int8 runs at ~70–96 ms per 10 s of audio on M-series.
- Built-in language detection covers zh / en / ja / ko / yue if you happen to drop in a non-Chinese segment.
- 152 MB on disk — small enough to keep loaded indefinitely.
- Meta-tag stripping (`<|HAPPY|>`, `<|EMO_NEUTRAL|>`, etc.) is automatic, so you never see emotion markers leak into the output.

**Trade-offs**
- transcribe-rs path **does not** support hot-words biasing. If you need term-level priors, switch the preset to Customized and pick `sense-voice-small-sherpa` (the sherpa-onnx variant, ~242 MB) — that path honours `hotwords_boost`.
- Single-pass decoding; no streaming partial output for long recordings. SenseVoice is fast enough that the gap is small in practice but visible on >30 s clips.

**Best for:** Chinese-first writing, journaling, WeChat / Slack replies, blog drafting, Obsidian capture.

### 2. Multilingual Offline (`multilingual_offline`)

| Layer | Setting |
|---|---|
| Engine | FunASR-Nano (sherpa-onnx path) |
| Language | `auto` |
| Punctuation | CT-Transformer-Punc enabled |
| Hot-words boost | Honoured via `OfflineRecognizerConfig.hotwords_score` |

**Benchmark results** (v0.8.6, 2026-05-02, 38 clips, 331 s audio, M-series, Tauri runtime):
P50 = **3510 ms** | P95 = **11164 ms** | punc density = 0.0852 | errors = 0  
These include a heavy cold-start penalty (FunASR-Nano's LLM decoder does OnnxRuntime JIT compilation
on first inference). Steady-state P50/P95 pending re-run with v0.8.6 single-spawn_blocking fix.

**Strengths**
- Strong multilingual decoder (LLM-style): handles code-mixed Chinese + English in a single utterance better than the other two presets.
- Fully offline. No network fall-back, no Apple cloud detour.
- sherpa-onnx + ONNX Runtime CPU; thread count auto-tuned to `min(physical_cores, 6)` so it stays off M-series efficiency cores.

**Trade-offs**
- ~1.0 GB model. Slower first-token latency than SenseVoice (typically 2-3× the wall time on the same clip).
- CoreML acceleration **deliberately not enabled** for the LLM decoder — sherpa-onnx issue #2910 shows it regresses RTF when the KV-cache repeatedly falls back to CPU. If you ever flip the provider to CoreML, expect a slowdown.

**Best for:** technical dictation that mixes Chinese + English jargon ("…我们用 ChatGPT 做 a brainstorming session…"), translator workflows, anyone whose threat model excludes any cloud round-trip.

### 3. Apple Native (`apple_native`)

| Layer | Setting |
|---|---|
| Engine | Apple Speech (`SFSpeechRecognizer` via Swift FFI) |
| Language | `auto` (resolved from `tauri_plugin_os::locale()` at load time) |
| On-device first | `apple_speech_require_on_device = true` (auto-falls-back to network if unavailable) |
| Punctuation | CT-Transformer-Punc enabled |
| Contextual hints | Custom words are forwarded as `addContextualStrings` to the recognizer |

**Benchmark results** (v0.8.5, 2026-05-02, run_bench.py, 38 clips, 331 s audio, M-series):
P50 = **788 ms** | P95 = **1788 ms** | punc density = 0.0082 (v0.8.5, CT-Punc bypassed) | errors = 4 (silence/noise)

> **v0.8.6 note:** CT-Punc now fires on Apple Native when recognised text contains CJK characters
> (CJK content-based fallback added in v0.8.6). Expected punc density ~0.04–0.06 for Chinese
> speech. Tauri runtime bench for apple_native is pending (blocked by SFSpeechRecognizer first-use
> hang with `requiresOnDeviceRecognition = true`).

**Strengths**
- Lowest first-token latency for short commands — Apple's recognizer is tuned for one-shot dictation buffers.
- Zero extra model weight; uses the system speech assets that Apple already ships.
- Streaming partial results render into the overlay window in real time (other presets only show the final transcript).
- Graceful degradation: returns `No speech detected` error on silence/noise rather than hallucinating.

**Trade-offs**
- Apple Speech does not output Chinese punctuation by itself. **Chinese punctuation is entirely supplied by the CT-Punc layer** — make sure `punc_zh_enabled` stays on. In v0.8.6, the CT-Punc layer fires whenever the recognised text contains any CJK ideograph, regardless of locale metadata (the v0.8.5 bug where `auto`+non-zh app_language skipped the punc layer is fixed).
- On-device locale availability is opaque: if the user hasn't downloaded the Chinese dictation model in System Settings, the recognizer silently falls back to network. Toggle `apple_speech_require_on_device = false` if you want to skip the on-device attempt entirely.
- macOS only. The preset is filtered out of the UI on Linux/Windows.

**Best for:** terminal commands, Slack one-liners, "open …", "search …" — anything where you stop talking after a phrase.

---

## How presets compose with the rest of Handy+

Presets are a **snapshot** mechanism, not a runtime fallback chain. When you click *Apply*, the preset writes its values straight into the underlying independent settings (`selected_model`, `selected_language`, `punc_zh_enabled`, …) and sets `active_preset_id`. Pipeline code keeps reading the independent settings, so behaviour is identical to manually configuring those fields.

If you later change any preset-controlled field (model dropdown, language picker, CT-Punc toggle), the active preset *detaches* — `active_preset_id` is cleared and the cards show a small "Customized" hint. Re-applying the preset re-snaps everything.

This composes cleanly with the other features added in v0.8.3-handy-plus:
- **Power Mode profiles** can override `selected_model` per app — gated by the new `profile_hot_swap_engine` master toggle (off by default, costs 1-3 s on stop). When enabled, the resolved engine for a recording can differ from the global preset's engine.
- **History retranscription** (the gear icon next to each entry) lets you replay any saved WAV through any model + language combination, regardless of the active preset.
- **Post-process chain** (`post_process_chain`) is preset-agnostic — chain steps run on the transcribed text after the preset's engine produces it.
- **Diary archival** (`diary_dir`) is also preset-agnostic; it inspects the raw transcription regardless of which engine produced it.

---

## Evaluating presets on your own audio

`benchmark/` ships a Tauri command that runs a preset over a directory of WAVs and writes a structured report.

```text
benchmark/
├── dataset/
│   ├── README.md           ← drop your *.wav files here (16 kHz mono recommended)
│   └── reference.txt       ← optional, one ground-truth line per wav (alphabetical wav order)
├── results/
│   └── <preset_id>_<ts>.json
├── eval.py                 ← markdown report + WER (jiwer if installed) + latency histogram
└── README.md
```

Workflow:
1. Drop ≥ 50 WAVs into `benchmark/dataset/` (handy already records every transcription into `~/Library/Application Support/com.pais.handy/recordings/` — a quick `cp` + sort yields a real-world dataset).
2. Optionally write a `reference.txt` for word-error-rate measurement.
3. Run `bun run tauri dev`, open the dev console, and invoke the benchmark for each preset:
   ```js
   await __TAURI__.core.invoke('run_asr_benchmark', {
     presetId: 'chinese_balanced',
     datasetDir: '/abs/path/handy/benchmark/dataset',
     outputDir: '/abs/path/handy/benchmark/results',
   });
   ```
4. `python benchmark/eval.py --compare benchmark/results/*.json` to produce a Markdown comparison table + latency histogram PNG.

The report JSON `summary` object tracks:

| Field | Meaning |
|---|---|
| `p50_latency_ms` / `p95_latency_ms` | Percentiles across **all** items (includes cold-start) |
| `cold_start_latency_ms` | Latency of the very first WAV — includes OnnxRuntime JIT, CT-Punc OnceCell init, etc. |
| `steady_p50_latency_ms` / `steady_p95_latency_ms` | Percentiles of items 2+ — **use these for real-world interactive latency estimates** |
| `total_audio_seconds` | Total audio duration processed |
| `punctuation_density` | Chinese/general punctuation chars / total chars |

> **Reading the numbers**: if `cold_start_latency_ms` is 3× higher than `steady_p50_latency_ms`,
> the first-inference JIT penalty is large — but users only pay it once per launch (or after model
> unload due to idle timeout). Steady-state P50 reflects what they feel for every subsequent
> press-and-release.

Add reference text to `reference.txt` (one ground-truth line per wav, alphabetical order) to also compute WER and per-term recall for hot-words validation.

---

## Decision tree

```
Are you dictating in Chinese ≥ 80 % of the time?
  ├─ Yes → Chinese Balanced
  └─ No → Do you mix languages within a single sentence?
            ├─ Yes → Multilingual Offline
            └─ No → Are most utterances < 5 seconds (commands, snippets)?
                      ├─ Yes → Apple Native (macOS only) / Chinese Balanced (everywhere else)
                      └─ No → Multilingual Offline if you need full offline guarantees,
                              otherwise stick with Chinese Balanced
```

---

## Known issues / non-goals

- The presets do **not** auto-download missing models. Apply will trigger a download via the existing model manager UI; first apply on a fresh install can take a minute on a slow network.
- Streaming partial-result rendering is currently Apple Speech only. SenseVoice and FunASR-Nano emit one final transcript at end-of-utterance. (See handoff: this is the "B. preheat + simulated streaming" item that was deliberately deferred.)
- CoreML execution provider is intentionally not used for FunASR-Nano. Do not toggle it back on without re-running `benchmark/eval.py` against issue #2910 reproduction.
- Hot-words biasing is engine-bound: only the sherpa-onnx variants honour `hotwords_boost`. The default Chinese Balanced preset uses transcribe-rs which ignores hot-words; switch to `sense-voice-small-sherpa` if biasing matters.

---

## Reference

- Settings struct: `src-tauri/src/settings.rs` (`AsrPreset`, `default_asr_presets`, `active_preset_id`, `profile_hot_swap_engine`, `punc_zh_enabled`, `hotwords_boost`, `diary_dir`, `post_process_chain`)
- Apply / detach commands: `src-tauri/src/commands/asr_presets.rs`
- Punctuation layer: `src-tauri/src/audio_toolkit/punc_zh.rs`
- SenseVoice meta filter: `src-tauri/src/audio_toolkit/sense_voice_filter.rs`
- Profile resolver: `src-tauri/src/profile_resolver.rs` (now also resolves `selected_model`)
- Benchmark command: `src-tauri/src/commands/benchmark.rs`
- Frontend cards: `src/components/settings/models/AsrPresetCards.tsx`
- Frontend retranscribe modal: `src/components/settings/history/RetranscribeModal.tsx`
