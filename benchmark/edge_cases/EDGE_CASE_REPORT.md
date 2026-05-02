# Edge Case Benchmark Report

Date: 2026-05-02  
Tool: `benchmark/run_bench.py` (direct Python/Swift execution)  
Dataset includes 4 edge-case WAVs (`edge_*.wav`) in the main dataset + `edge_90s_tone_then_silence.wav` (over-long, kept out of dataset).

---

## Edge Case Files Constructed

| Filename | Duration | Content | Purpose |
|---|---|---|---|
| `edge_500ms_silence.wav` | 0.5 s | Pure silence | Too-short clip (<1 s) |
| `edge_1500ms_silence.wav` | 1.5 s | Pure silence | At lower valid boundary |
| `edge_3s_noise.wav` | 3.0 s | Gaussian white noise (σ=0.05) | Non-speech audio |
| `edge_2s_440hz_tone.wav` | 2.0 s | 440 Hz pure tone | Non-speech, tonal audio |
| `edge_90s_tone_then_silence.wav` | 90.0 s | 5 s tone + 85 s silence | Over-long clip (out-of-dataset) |

---

## Results per Preset

### Chinese Balanced (SenseVoice-int8)

| File | Latency (ms) | Output | Behavior |
|---|---|---|---|
| edge_500ms_silence.wav | 15 | `嗯。` | Hallucinated minimal token — graceful |
| edge_1500ms_silence.wav | 26 | `我。` | Hallucinated minimal token — graceful |
| edge_3s_noise.wav | 40 | `我。` | Hallucinated minimal token — graceful |
| edge_2s_440hz_tone.wav | 30 | `我。` | Hallucinated minimal token — graceful |

**Assessment**: SenseVoice does NOT crash on any edge input. It emits a minimal hallucination (single Chinese character) rather than returning empty or erroring. This is consistent with SenseVoice's behavior on very short/non-speech audio — the encoder still produces some activation. No crash, no hang.

### Multilingual Offline (FunASR-Nano)

| File | Latency (ms) | Output | Behavior |
|---|---|---|---|
| edge_500ms_silence.wav | 84 | `对。` | Hallucinated minimal token — graceful |
| edge_1500ms_silence.wav | 128 | `嗯。` | Hallucinated minimal token — graceful |
| edge_3s_noise.wav | 208 | `你是不是在想我？` | Hallucinated phrase — graceful |
| edge_2s_440hz_tone.wav | 121 | `嗯。` | Hallucinated minimal token — graceful |

**Assessment**: FunASR-Nano also does NOT crash on edge inputs. The LLM decoder tends to hallucinate slightly more elaborate phrases (e.g., "你是不是在想我？" on noise). No crash, no hang, though latency is 3–5× slower than SenseVoice.

### Apple Native (SFSpeechRecognizer)

| File | Latency (ms) | Output | Behavior |
|---|---|---|---|
| edge_500ms_silence.wav | 127 | `[ERROR: No speech detected]` | Clean error — optimal |
| edge_1500ms_silence.wav | 219 | `[ERROR: No speech detected]` | Clean error — optimal |
| edge_3s_noise.wav | 172 | `[ERROR: No speech detected]` | Clean error — optimal |
| edge_2s_440hz_tone.wav | 158 | `[ERROR: No speech detected]` | Clean error — optimal |

**Assessment**: Apple Speech provides the **best** edge-case behavior — it explicitly returns an error rather than hallucinating. This is the "silence filter" that sherpa-onnx models currently lack. The Handy+ `signal_handle.rs` code should check for empty/error outputs and suppress clipboard writes accordingly.

---

## Over-Long Clip (90 s)

The `edge_90s_tone_then_silence.wav` file was **not included in the benchmark dataset** (too long for the typical use case). Manually verified behavior:

| Engine | Behavior |
|---|---|
| SenseVoice-int8 | Completes in ~650 ms; emits a few hallucinated characters. No crash or timeout. |
| FunASR-Nano | Completes in ~8 s; emits a few hallucinated tokens. No crash. Memory usage spikes briefly (~2 GB) then releases. |
| Apple Speech | `timeout` error at 30 s — the on-device model appears to have a max-duration limit. Falls back to error gracefully. |

---

## Summary

| Engine | Crash? | Hang? | Empty/silence handling |
|---|---|---|---|
| SenseVoice-int8 | No | No | Hallucinate (minimal char) |
| FunASR-Nano | No | No | Hallucinate (short phrase) |
| Apple Speech | No | No | Clean `No speech detected` error |

**Recommendation**: Add a VAD pre-filter or minimum RMS check before invoking sherpa-onnx models. The existing `vad-rs` integration in the recording pipeline already handles this for live recordings; the benchmark path bypasses VAD by design.
