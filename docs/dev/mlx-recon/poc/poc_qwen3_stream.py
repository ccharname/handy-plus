#!/usr/bin/env python3
"""
PoC: Qwen3-ASR-0.6B streaming transcription (smallest model).
"""
import time
import sys
import os

AUDIO = "/tmp/qwen3_bench_zh.wav"
MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"

print(f"[PoC] Python {sys.version}")

if not os.path.exists(AUDIO):
    print(f"[ERROR] Audio file not found: {AUDIO}")
    sys.exit(1)

print(f"[PoC] Loading mlx-audio STT...")
try:
    from mlx_audio.stt.generate import load_model, generate_transcription
    print("[PoC] Import OK")
except ImportError as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

print(f"[PoC] Loading {MODEL_ID} ...")
t_load = time.time()
try:
    model = load_model(MODEL_ID)
    print(f"[PoC] Loaded in {time.time()-t_load:.2f}s")
except Exception as e:
    print(f"[ERROR] Load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test stream_transcribe if available
print("\n[PoC] Testing stream_transcribe()...")
t_start = time.time()
first = None
try:
    for i, token in enumerate(model.stream_transcribe(AUDIO, language="Chinese")):
        if first is None:
            first = time.time()
            print(f"[PoC] TTFT: {(first-t_start)*1000:.0f}ms")
        elapsed = time.time() - t_start
        print(f"[t={elapsed:.2f}s] {repr(token)}", flush=True)
    print(f"[PoC] Done in {time.time()-t_start:.2f}s")
except AttributeError:
    print("[PoC] stream_transcribe() not available, trying generate(stream=True)...")
    try:
        for result in model.generate(AUDIO, stream=True):
            if first is None:
                first = time.time()
                print(f"[PoC] TTFT: {(first-t_start)*1000:.0f}ms")
            print(f"[t={time.time()-t_start:.2f}s] {repr(result)}", flush=True)
    except Exception as e2:
        print(f"[ERROR] {e2}")
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
