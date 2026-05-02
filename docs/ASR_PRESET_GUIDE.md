# Handy+ ASR Preset Guide

Handy+ ships three pre-tuned ASR (automatic speech recognition) configurations, exposed in **Settings → Models → ASR Presets**. Each preset is a one-click bundle of {engine, language, punctuation layer, hot-words wiring} that lets you switch the speech stack without poking individual settings.

This document explains what each preset is good for, the trade-offs between them, and how to evaluate them on your own audio using `benchmark/`.

---

## TL;DR — Picking a preset

| Use case | Preset | Why |
|---|---|---|
| Mostly Chinese dictation (writing, chat, notes) | **Chinese Balanced** | SenseVoice is fast (≈70 ms / 10 s clip), CT-Transformer-Punc adds standard Chinese punctuation, low memory footprint (~152 MB) |
| Multilingual mixed input (zh+en+ja+...) — must run offline | **Multilingual Offline** | FunASR-Nano LLM-decoder handles 8+ languages well; CT-Punc still patches Chinese punctuation |
| Fast English short commands, latency-critical, system-language-driven | **Apple Native** | SFSpeechRecognizer first-token latency is the lowest; CT-Punc kicks in only when system locale is Chinese |
| Anything else / power user | **Advanced (Customized)** — keep the per-field controls below the preset cards |

If you don't know which one to start with: **Chinese Balanced is the v0.8.3-handy-plus.6 default**. Apply it once and start dictating.

### Benchmark numbers (v0.8.5, M-series, 38 clips, 331 s audio)

> Measured 2026-05-02 with `benchmark/run_bench.py` on 38 real + synthetic WAVs  
> (5 user recordings + 5 SenseVoice test wavs + 24 FunASR-Nano test wavs + 4 edge-case clips).

| Metric | Chinese Balanced | Multilingual Offline | Apple Native |
|---|---|---|---|
| Engine | SenseVoice-int8 | FunASR-Nano | Apple Speech |
| P50 latency (ms) | **96** | 581 | 788 |
| P95 latency (ms) | **280** | 1955 | 1788 |
| Punctuation density | 0.0580 | 0.0667 | 0.0082 |
| Errors (38 clips) | 0 | 0 | 4 (silence/noise) |

Key takeaways:
- **Chinese Balanced is 6× faster at P50** than Multilingual Offline; the gap widens at P95 (~7×).
- **Apple Native** has the lowest punc density — Apple Speech produces minimal punctuation natively; the CT-Punc layer adds some but the overall density is still ~7× lower than sherpa-onnx models.
- **Apple Native gracefully errors** on silence/noise clips (`No speech detected`) — clean degradation.
- **FunASR-Nano** shines on Chinese quality (higher punc density = more complete sentences) and handles code-mixed content but pays a latency tax.

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

**Benchmark results** (2026-05-02, 38 clips, 331 s audio, M-series):
P50 = **96 ms** | P95 = **280 ms** | punc density = 0.0580 | errors = 0

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

**Benchmark results** (2026-05-02, 38 clips, 331 s audio, M-series):
P50 = **581 ms** | P95 = **1955 ms** | punc density = 0.0667 | errors = 0

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

**Benchmark results** (2026-05-02, 38 clips, 331 s audio, M-series):
P50 = **788 ms** | P95 = **1788 ms** | punc density = 0.0082 | errors = 4 (silence/noise)

**Strengths**
- Lowest first-token latency for short commands — Apple's recognizer is tuned for one-shot dictation buffers.
- Zero extra model weight; uses the system speech assets that Apple already ships.
- Streaming partial results render into the overlay window in real time (other presets only show the final transcript).
- Graceful degradation: returns `No speech detected` error on silence/noise rather than hallucinating.

**Trade-offs**
- Apple Speech does not output Chinese punctuation by itself. **Chinese punctuation is entirely supplied by the CT-Punc layer** — make sure `punc_zh_enabled` stays on.
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

The report tracks: p50/p95 wall-clock latency, total audio seconds, punctuation density (Chinese punctuation chars / total chars), per-item hypothesis. Add reference text to also compute WER and per-term recall for hot-words validation.

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
