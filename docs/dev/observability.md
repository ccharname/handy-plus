# Handy Observability

Full-pipeline instrumentation for the handy-plus STT pipeline.  Every hotkey→output cycle is tagged with a ULID request id and emits one JSON event per stage into `~/Library/Logs/Handy/handy.jsonl` (daily rotation).

## Pipeline diagram

```
Hotkey press
    │
    ▼ t0_hotkey ──────── rdev/handy-keys keydown → TranscriptionCoordinator dispatch
    │
    ▼ t1_audio_capture ─ cpal stream.play() → first frame received
    │
    ▼ t2_recording ───── continuous recording (VAD + resample inline in consumer thread)
    │                    emits: sample_count, audio_duration_ms
    │
    ├─ t3_vad ─────────── Silero VAD pre-filter (silence gate before inference)
    │                     emits: vad_ms, voice_frames, total_frames, speech_ratio, gate
    │
    ├─ t4_resample ────── rubato 16kHz resample (measured inline with t2)
    │
    ▼ t5_inference ────── model inference
    │   ├─ SenseVoice:   encoder → decoder (sherpa-onnx)
    │   └─ Qwen3-MLX:    prefill → decode → streaming_chunk[]
    │   emits: inference_ms, audio_duration_ms, rtf, transcript_char_count
    │          (+ transcript if --log-transcripts)
    │
    ▼ t6_postprocess ──── ITN / punc_dedup / hotword injection / OpenCC / LLM post-proc
    │                     emits: post_processed (bool), final_char_count
    │
    ▼ t7_output ─────────clipboard set + paste dispatch
    │                     emits: clipboard_set_ms
    │
    ▼ total ─────────────end-to-end (t0 keydown → paste complete)
                          emits: end_to_end_ms, outcome
```

## Stage definitions

| ID | Boundary | Key metrics |
|---|---|---|
| `t0_hotkey` | rdev keydown → coordinator dispatch complete | `dispatch_ms` (= duration_ms) |
| `t1_audio_capture` | `cpal stream.play()` returns | `capture_open_ms` |
| `t2_recording` | Recording session (VAD+resample inline) | `audio_duration_ms`, `sample_count` |
| `t3_vad` | Silero silence-gate check before inference | `vad_ms`, `voice_frames`, `speech_ratio`, `gate` |
| `t4_resample` | rubato 16kHz (currently measured inline with t2) | `audio_duration_ms` |
| `t5_inference` | Model inference start → result returned | `inference_ms`, `rtf`, `transcript_char_count` |
| `t6_postprocess` | ITN + dedup + hotwords + LLM post-proc | `post_processed`, `final_char_count` |
| `t7_output` | Clipboard set + paste dispatch | `clipboard_set_ms` |
| `total` | t0 keydown → paste complete | `end_to_end_ms`, `outcome` |

### Outcome values

- `ok` — stage completed successfully
- `cancelled` — user cancelled (recording produced no samples, or user pressed cancel)
- `error` — unhandled error / exception
- `timeout` — stage exceeded deadline (reserved; not yet emitted)

## Log format

Each line in `handy.jsonl` is a JSON object from `tracing-subscriber`:

```json
{
  "timestamp": "2026-05-04T10:23:45.123Z",
  "level": "INFO",
  "fields": {
    "request_id": "01HX...",
    "stage": "t5_inference",
    "outcome": "ok",
    "duration_ms": 312.4,
    "ts_unix_ms": 1746350625123,
    "extra": "inference_ms=312 audio_duration_ms=8500 rtf=0.037 transcript_char_count=42",
    "message": "stage_event"
  },
  "target": "handy_app_lib::observability"
}
```

### Privacy boundary

- Default: `transcript` field is **never** written. Only `transcript_char_count` is emitted.
- Override: launch with `--log-transcripts` to include raw transcript text in t5 events.

## CLI: handy-logs

Located at `scripts/handy-logs.sh` (Python 3, stdlib only, chmod +x).

### Cold-start investigation

```bash
# 1. See if t0 or t1 is the bottleneck
./scripts/handy-logs.sh percentiles --stage t0_hotkey --since 7d
./scripts/handy-logs.sh percentiles --stage t1_audio_capture --since 7d

# 2. Full waterfall for the last slow run
./scripts/handy-logs.sh slowest --stage total --top 1
./scripts/handy-logs.sh trace <req_id from above>
```

### Inference bottleneck

```bash
# Compare RTF across both presets
./scripts/handy-logs.sh percentiles --stage t5_inference --metric rtf --since 7d

# SenseVoice specific
./scripts/handy-logs.sh percentiles --stage t5_inference --preset chinese_balanced --metric rtf
```

### Memory leak check (50-iter)

Run `handy-logs breakdown --since 1h` after 50 consecutive recordings and look for monotonic growth in any stage's mean_ms.

### SLA hard gates

```bash
./scripts/handy-logs.sh assert --stage t0_hotkey --p50-max 30 --p99-max 80
./scripts/handy-logs.sh assert --stage t1_audio_capture --p50-max 100 --p99-max 250
./scripts/handy-logs.sh assert --stage t3_vad --p50-max 50 --p99-max 120
./scripts/handy-logs.sh assert --stage t4_resample --p50-max 30 --p99-max 80
./scripts/handy-logs.sh assert --stage t5_inference --preset chinese_balanced --metric rtf --p50-max 0.30 --p99-max 0.50
./scripts/handy-logs.sh assert --stage t5_inference --preset qwen3_mlx --metric first_token_ms --p50-max 400 --p99-max 800
./scripts/handy-logs.sh assert --stage t5_inference --preset qwen3_mlx --metric rtf --p50-max 0.40 --p99-max 0.70
./scripts/handy-logs.sh assert --metric startup_ms --p50-max 1500 --p99-max 2500
```

## End-to-end verification recipe (requires zheng to run manually)

1. Build and launch the app normally.
2. Trigger one recording with the hotkey — say ~5 seconds of speech.
3. After paste completes:
   ```bash
   # Confirm 9 stage events were written for the run
   tail -n 50 ~/Library/Logs/Handy/handy.jsonl | python3 -c "
   import sys, json
   lines = [json.loads(l) for l in sys.stdin if l.strip()]
   stages = [l.get('fields', l).get('stage') for l in lines if l.get('fields', l).get('stage')]
   print('stages found:', stages[-10:])
   "

   # Get the last request_id
   REQ=$(python3 -c "
   import json, pathlib
   lines = pathlib.Path.home().joinpath('Library/Logs/Handy/handy.jsonl').read_text().strip().splitlines()
   for line in reversed(lines):
       try:
           obj = json.loads(line)
           f = obj.get('fields', obj)
           if f.get('stage') == 'total':
               print(f.get('request_id', ''))
               break
       except: pass
   ")
   echo "Last req: $REQ"

   # View waterfall
   ./scripts/handy-logs.sh trace "$REQ"

   # Stage breakdown
   ./scripts/handy-logs.sh breakdown

   # Run SLA assertions (will skip if < 1 sample)
   ./scripts/handy-logs.sh assert --stage t0_hotkey --p50-max 30 --p99-max 80
   ```

## Notes on t3/t4 inline measurement

VAD (t3) and resample (t4) run frame-by-frame inside the audio consumer thread and are not separated from the recording wall-clock time (t2). The current implementation emits placeholder events with `duration_ms=0` and `note=inline_with_recording` for t3/t4. Real per-frame timing is tracked as a M4 improvement item.

The meaningful t3 measurement for the SenseVoice preset is the silence_gate pre-check, which *does* emit a real `vad_ms` value (Silero runs on the full captured buffer before inference).
