#!/usr/bin/env python3
"""
PoC: Voxtral Realtime streaming transcription
Measures TTFT and streaming token cadence.
"""
import time
import sys
import os

AUDIO = "/tmp/qwen3_bench_zh.wav"
MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"

print(f"[PoC] Python {sys.version}")
print(f"[PoC] Audio: {AUDIO}")

if not os.path.exists(AUDIO):
    print(f"[ERROR] Audio file not found: {AUDIO}")
    sys.exit(1)

try:
    import scipy.io.wavfile as wav
    sr, data = wav.read(AUDIO)
    duration = len(data) / sr
    print(f"[PoC] Audio: {duration:.1f}s @ {sr}Hz, shape={data.shape}")
except Exception as e:
    print(f"[WARN] Could not read WAV: {e}")

print("\n[PoC] Loading mlx-audio STT module...")
t0 = time.time()

try:
    from mlx_audio.stt.generate import load_model, generate_transcription
    print(f"[PoC] Import OK in {time.time()-t0:.2f}s")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

print(f"\n[PoC] Loading model: {MODEL_ID}")
print("[PoC] (This will download ~2-4GB on first run...)")
t_load = time.time()

try:
    model = load_model(MODEL_ID)
    print(f"[PoC] Model loaded in {time.time()-t_load:.2f}s")
    print(f"[PoC] Model type: {type(model)}")
    print(f"[PoC] generate() params: {list(__import__('inspect').signature(model.generate).parameters.keys())}")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[PoC] === STREAMING TEST (delay=480ms) ===")
t_start = time.time()
first_token_time = None
deltas = []

try:
    for i, result in enumerate(model.generate(AUDIO, stream=True, transcription_delay_ms=480)):
        now = time.time()
        if first_token_time is None:
            first_token_time = now
            ttft = first_token_time - t_start
            print(f"[PoC] TTFT: {ttft*1000:.0f}ms")

        if hasattr(result, 'text'):
            delta = result.text
            is_final = getattr(result, 'is_final', False)
            lang = getattr(result, 'language', '?')
        else:
            delta = str(result)
            is_final = False
            lang = '?'

        deltas.append(delta)
        elapsed = now - t_start
        print(f"  [t={elapsed:.2f}s i={i}] is_final={is_final} lang={lang} delta={repr(delta)[:80]}", flush=True)

        if is_final:
            print(f"[PoC] Final result received at t={elapsed:.2f}s")
            break

    total_time = time.time() - t_start
    full_text = ''.join(deltas)
    print(f"\n[PoC] === STREAM RESULTS ===")
    print(f"Total time: {total_time:.2f}s")
    if first_token_time:
        print(f"TTFT: {(first_token_time - t_start)*1000:.0f}ms")
    print(f"Iterations: {len(deltas)}")
    print(f"Full text: {repr(full_text[:200])}")

except Exception as e:
    print(f"[ERROR] Streaming failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[PoC] === BATCH BASELINE ===")
t_batch = time.time()
try:
    result = model.generate(AUDIO, stream=False)
    batch_time = time.time() - t_batch
    if hasattr(result, 'text'):
        print(f"Text: {repr(result.text[:200])}")
        print(f"Batch time: {batch_time:.2f}s")
        if hasattr(result, 'generation_tokens'):
            print(f"Tokens: {result.generation_tokens}")
    else:
        print(f"Result: {repr(str(result)[:200])}")
        print(f"Batch time: {batch_time:.2f}s")
except Exception as e:
    print(f"[ERROR] Batch failed: {e}")
    import traceback
    traceback.print_exc()
