# Handy+ ASR Benchmark

Programmatic benchmark for evaluating and comparing the 3 built-in ASR presets:

| Preset ID              | Engine          | Language |
| ---------------------- | --------------- | -------- |
| `chinese_balanced`     | SenseVoice int8 | zh-Hans  |
| `multilingual_offline` | FunASR-Nano     | auto     |
| `apple_native`         | Apple Speech    | auto     |

---

## 1. Prepare test data

```
benchmark/
└── dataset/
    ├── README.md          ← this file
    ├── test01.wav
    ├── test02.wav
    ├── ...
    └── reference.txt      ← optional, one reference per wav (sorted order)
```

See [`dataset/README.md`](dataset/README.md) for detailed requirements and how to
export 50 samples from your Handy recording history.

---

## 2. Run the benchmark (via Tauri debug console)

1. Start the app:
   ```bash
   bun run tauri dev
   ```

2. Open the debug console with **Cmd+Shift+D** (macOS) or **Ctrl+Shift+D**
   (Windows/Linux).

3. Invoke the command from the console (replace paths as needed):

   ```javascript
   // Single preset
   await __TAURI__.core.invoke('run_asr_benchmark', {
     presetId: 'chinese_balanced',
     datasetDir: '/Users/you/Developer/handy/benchmark/dataset',
     outputDir:  '/Users/you/Developer/handy/benchmark/results',
   })

   // Run all three presets sequentially
   const presets = ['chinese_balanced', 'multilingual_offline', 'apple_native'];
   for (const presetId of presets) {
     const report = await __TAURI__.core.invoke('run_asr_benchmark', {
       presetId,
       datasetDir: '/Users/you/Developer/handy/benchmark/dataset',
       outputDir:  '/Users/you/Developer/handy/benchmark/results',
     });
     console.log(`${presetId}: p50=${report.summary.p50_latency_ms}ms p95=${report.summary.p95_latency_ms}ms`);
   }
   ```

4. JSON reports are written to `benchmark/results/` automatically.

**Progress events** are emitted as `bench-progress` on the Tauri event bus:
```javascript
await __TAURI__.event.listen('bench-progress', (e) => console.log(e.payload));
```

**Note on timing:** Each run loads the model, then transcribes every WAV file in
`dataset_dir` sequentially. Expect 2–10 minutes for 50 files depending on the model
and hardware.

---

## 3. Evaluate results

```bash
# Single report — prints summary + sample hypotheses
python benchmark/eval.py benchmark/results/chinese_balanced_2026-05-01T22-00-00Z.json

# With WER/CER (requires reference.txt in the same order as sorted WAV filenames)
python benchmark/eval.py \
    benchmark/results/chinese_balanced_*.json \
    --reference benchmark/dataset/reference.txt

# Compare three presets side by side
python benchmark/eval.py --compare \
    benchmark/results/chinese_balanced_*.json \
    benchmark/results/multilingual_offline_*.json \
    benchmark/results/apple_native_*.json
```

### Optional dependencies

| Package      | Purpose                          | Install                  |
| ------------ | -------------------------------- | ------------------------ |
| `jiwer`      | Word Error Rate calculation      | `pip install jiwer`      |
| `matplotlib` | Latency histogram PNG            | `pip install matplotlib` |

Both are optional — the evaluator works without them (CER uses a built-in edit
distance implementation; histograms are skipped).

---

## 4. Reading the comparison report

The comparison table (`benchmark/results/comparison_<timestamp>.md`) looks like:

```
| Metric           | Chinese Balanced | Multilingual Offline | Apple Native |
| ---------------- | ---------------- | -------------------- | ------------ |
| Model            | sense-voice-int8 | funasr-nano          | apple-speech |
| P50 latency (ms) | 142              | 210                  | 88           |
| P95 latency (ms) | 488              | 620                  | 340          |
| Punc density     | 0.0780           | 0.0510               | 0.0320       |
```

Key metrics:

- **P50 latency** — median transcription time per utterance (lower is better)
- **P95 latency** — worst-case for 95 % of utterances (lower is better)
- **Punc density** — fraction of output characters that are punctuation (higher
  means richer sentence boundary annotation)
- **CER / WER** — only present when `reference.txt` is supplied

---

## 5. JSON report schema

```json
{
  "preset_id": "chinese_balanced",
  "preset_name": "Chinese Balanced",
  "model_id": "sense-voice-int8",
  "language": "zh-Hans",
  "punc_zh_enabled": true,
  "timestamp": "2026-05-01T22:00:00Z",
  "items": [
    {
      "wav": "test01.wav",
      "hypothesis": "今天天气真不错。",
      "latency_ms": 87,
      "punc_count": 1,
      "char_count": 8
    }
  ],
  "summary": {
    "total_items": 50,
    "total_audio_seconds": 312.4,
    "p50_latency_ms": 142,
    "p95_latency_ms": 488,
    "punctuation_density": 0.078
  }
}
```
