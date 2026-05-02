#!/usr/bin/env python3
"""
Handy+ ASR benchmark runner — direct Python/Swift execution.

Runs all three presets (chinese_balanced, multilingual_offline, apple_native)
directly, bypassing the Tauri runtime:
- chinese_balanced:   sherpa-onnx SenseVoice (zh-Hans)
- multilingual_offline: sherpa-onnx FunASR-Nano
- apple_native:       SFSpeechRecognizer via compiled Swift helper

Usage:
    python benchmark/run_bench.py [--presets chinese_balanced,...] \
                                  [--dataset benchmark/dataset] \
                                  [--output benchmark/results]

Requirements:
    pip install sherpa-onnx
    benchmark/apple_speech_bench  (pre-compiled Swift binary — run make_apple_bench.sh once)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Punctuation counting (mirrors Rust benchmark.rs)
# ---------------------------------------------------------------------------
_PUNC_CHARS = set("。，、；：？！.,;:?!")


def count_punc(s: str) -> int:
    return sum(1 for c in s if c in _PUNC_CHARS)


# ---------------------------------------------------------------------------
# WAV reading
# ---------------------------------------------------------------------------

def read_wav_f32(path: str) -> tuple[list[float], int]:
    """Return (samples_f32, sample_rate)."""
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(n)

    if sw == 2:
        raw = struct.unpack(f"<{len(frames)//2}h", frames)
        samples = [s / 32768.0 for s in raw]
    elif sw == 4:
        raw = struct.unpack(f"<{len(frames)//4}i", frames)
        samples = [s / 2147483648.0 for s in raw]
    elif sw == 1:
        raw = struct.unpack(f"<{len(frames)}B", frames)
        samples = [(s - 128) / 128.0 for s in raw]
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    # Mix down to mono if needed
    if ch == 2:
        samples = [(samples[i] + samples[i + 1]) / 2 for i in range(0, len(samples) - 1, 2)]

    return samples, rate


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

class SenseVoiceEngine:
    """SenseVoice via sherpa-onnx Python bindings."""

    def __init__(self, model_dir: str, language: str = "zh"):
        import sherpa_onnx
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=os.path.join(model_dir, "model.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=4,
            language=language,
            use_itn=True,
            debug=False,
        )

    def transcribe(self, samples: list[float], rate: int) -> tuple[str, int]:
        """Returns (text, latency_ms)."""
        t0 = time.monotonic()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self.recognizer.decode_stream(stream)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return stream.result.text, latency_ms


class FunASRNanoEngine:
    """FunASR-Nano via sherpa-onnx Python bindings."""

    def __init__(self, model_dir: str):
        import sherpa_onnx
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=os.path.join(model_dir, "encoder_adaptor.int8.onnx"),
            llm=os.path.join(model_dir, "llm.int8.onnx"),
            embedding=os.path.join(model_dir, "embedding.int8.onnx"),
            tokenizer=os.path.join(model_dir, "Qwen3-0.6B"),
            num_threads=4,
            debug=False,
        )

    def transcribe(self, samples: list[float], rate: int) -> tuple[str, int]:
        t0 = time.monotonic()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self.recognizer.decode_stream(stream)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return stream.result.text, latency_ms


class AppleSpeechEngine:
    """Apple Speech via compiled Swift binary."""

    def __init__(self, swift_bin: str, locale: str = "auto"):
        self.swift_bin = swift_bin
        self.locale = locale
        if not os.path.exists(swift_bin):
            raise FileNotFoundError(f"Apple Speech binary not found: {swift_bin}")

    def _detect_locale(self, wav_path: str) -> str:
        """Use 'auto' locale heuristic — default to zh-Hans then en-US as fallback."""
        if self.locale != "auto":
            return self.locale
        return "zh-Hans"

    def transcribe(self, samples: list[float], rate: int, wav_path: str = "") -> tuple[str, int]:
        locale = self._detect_locale(wav_path)
        try:
            result = subprocess.run(
                [self.swift_bin, locale, wav_path],
                capture_output=True,
                text=True,
                timeout=45,
            )
            if result.returncode != 0:
                return f"[ERROR: rc={result.returncode}]", 0
            data = json.loads(result.stdout.strip())
            text = data.get("text", "")
            latency_ms = data.get("latency_ms", 0)
            err = data.get("error", "")
            if err:
                return f"[ERROR: {err}]", latency_ms
            return text, latency_ms
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]", 45000
        except Exception as exc:
            return f"[EXCEPTION: {exc}]", 0


# ---------------------------------------------------------------------------
# Preset definitions (mirror settings.rs default_asr_presets)
# ---------------------------------------------------------------------------

def make_engine(preset_id: str, models_base: str, swift_bin: str):
    """Create the appropriate engine for a given preset."""
    if preset_id == "chinese_balanced":
        model_dir = os.path.join(
            models_base, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        )
        return SenseVoiceEngine(model_dir, language="zh")

    elif preset_id == "multilingual_offline":
        model_dir = os.path.join(models_base, "sherpa-onnx-funasr-nano-int8-2025-12-30")
        return FunASRNanoEngine(model_dir)

    elif preset_id == "apple_native":
        return AppleSpeechEngine(swift_bin, locale="auto")

    else:
        raise ValueError(f"Unknown preset: {preset_id}")


PRESET_META = {
    "chinese_balanced": {
        "name": "Chinese Balanced",
        "model_id": "sense-voice-int8",
        "language": "zh-Hans",
        "punc_zh_enabled": True,
    },
    "multilingual_offline": {
        "name": "Multilingual Offline",
        "model_id": "funasr-nano",
        "language": "auto",
        "punc_zh_enabled": True,
    },
    "apple_native": {
        "name": "Apple Native",
        "model_id": "apple-speech",
        "language": "auto",
        "punc_zh_enabled": True,
    },
}


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(p / 100.0 * (len(sorted_vals) - 1) + 0.5)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run_preset(
    preset_id: str,
    dataset_dir: str,
    output_dir: str,
    models_base: str,
    swift_bin: str,
    progress: bool = True,
) -> dict:
    meta = PRESET_META[preset_id]

    wav_files = sorted(Path(dataset_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {dataset_dir}")

    print(f"\n{'='*60}")
    print(f"Preset: {meta['name']} ({preset_id})")
    print(f"Model:  {meta['model_id']}")
    print(f"Files:  {len(wav_files)}")
    print(f"{'='*60}")
    print("Loading engine... ", end="", flush=True)
    t_load = time.monotonic()
    engine = make_engine(preset_id, models_base, swift_bin)
    load_ms = int((time.monotonic() - t_load) * 1000)
    print(f"ready ({load_ms} ms)")

    items = []
    total_audio_seconds = 0.0
    errors = 0

    for i, wav_path in enumerate(wav_files):
        wav_name = wav_path.name
        if progress:
            print(f"  [{i+1}/{len(wav_files)}] {wav_name}...", end=" ", flush=True)

        try:
            samples, rate = read_wav_f32(str(wav_path))
        except Exception as exc:
            print(f"READ_ERROR: {exc}")
            items.append({
                "wav": wav_name,
                "hypothesis": f"[READ_ERROR: {exc}]",
                "latency_ms": 0,
                "punc_count": 0,
                "char_count": 0,
                "error": str(exc),
            })
            errors += 1
            continue

        audio_dur = len(samples) / rate
        total_audio_seconds += audio_dur

        try:
            if isinstance(engine, AppleSpeechEngine):
                text, latency_ms = engine.transcribe(samples, rate, wav_path=str(wav_path))
            else:
                text, latency_ms = engine.transcribe(samples, rate)
        except Exception as exc:
            text = f"[TRANSCRIBE_ERROR: {exc}]"
            latency_ms = 0
            errors += 1

        punc = count_punc(text)
        chars = len(text)

        if progress:
            preview = text[:60].replace("\n", " ") if text and not text.startswith("[") else text
            print(f"{latency_ms}ms | {preview}")

        items.append({
            "wav": wav_name,
            "hypothesis": text,
            "latency_ms": latency_ms,
            "punc_count": punc,
            "char_count": chars,
        })

    # Compute summary
    latencies = sorted([it["latency_ms"] for it in items if it["latency_ms"] > 0])
    total_punc = sum(it["punc_count"] for it in items)
    total_chars = sum(it["char_count"] for it in items)
    punc_density = total_punc / total_chars if total_chars > 0 else 0.0

    summary = {
        "total_items": len(items),
        "total_audio_seconds": round(total_audio_seconds, 3),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "punctuation_density": round(punc_density, 6),
        "errors": errors,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    report = {
        "preset_id": preset_id,
        "preset_name": meta["name"],
        "model_id": meta["model_id"],
        "language": meta["language"],
        "punc_zh_enabled": meta["punc_zh_enabled"],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": items,
        "summary": summary,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, f"{preset_id}_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"\nSummary: {len(items)} items | P50={summary['p50_latency_ms']}ms | "
          f"P95={summary['p95_latency_ms']}ms | punc_density={punc_density:.4f} | errors={errors}")
    print(f"Report: {report_path}")

    return {"report": report, "path": report_path}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Handy+ direct benchmark runner")
    parser.add_argument("--presets", default="chinese_balanced,multilingual_offline,apple_native",
                        help="Comma-separated preset IDs")
    parser.add_argument("--dataset", default="benchmark/dataset",
                        help="Dataset directory (default: benchmark/dataset)")
    parser.add_argument("--output", default="benchmark/results",
                        help="Output directory for JSON reports (default: benchmark/results)")
    parser.add_argument("--models-base",
                        default=str(Path.home() / "Library/Application Support/com.pais.handy/models"),
                        help="Models base directory")
    parser.add_argument("--swift-bin", default="",
                        help="Path to compiled apple_speech_bench binary")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval.py after benchmarks")
    args = parser.parse_args()

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    # Resolve paths
    repo_root = Path(__file__).parent.parent
    dataset_dir = str((repo_root / args.dataset).resolve()) if not os.path.isabs(args.dataset) else args.dataset
    output_dir = str((repo_root / args.output).resolve()) if not os.path.isabs(args.output) else args.output
    swift_bin = args.swift_bin or str(Path(__file__).parent / "apple_speech_bench")

    print("Handy+ ASR Benchmark Runner")
    print(f"Presets:  {presets}")
    print(f"Dataset:  {dataset_dir}")
    print(f"Output:   {output_dir}")
    print(f"Models:   {args.models_base}")

    wav_count = len(list(Path(dataset_dir).glob("*.wav"))) if Path(dataset_dir).exists() else 0
    if wav_count == 0:
        print(f"ERROR: No WAV files found in {dataset_dir}", file=sys.stderr)
        return 1
    print(f"WAV files: {wav_count}\n")

    result_paths = []
    for preset_id in presets:
        if preset_id not in PRESET_META:
            print(f"WARNING: Unknown preset '{preset_id}', skipping.", file=sys.stderr)
            continue
        try:
            result = run_preset(
                preset_id,
                dataset_dir,
                output_dir,
                args.models_base,
                swift_bin,
            )
            result_paths.append(result["path"])
        except Exception as exc:
            import traceback
            print(f"\nERROR running preset '{preset_id}': {exc}", file=sys.stderr)
            traceback.print_exc()

    if not result_paths:
        print("ERROR: No results produced.", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"Benchmarks complete: {len(result_paths)}/{len(presets)} presets succeeded")

    if not args.skip_eval and len(result_paths) >= 1:
        eval_script = str(Path(__file__).parent / "eval.py")
        cmd = [sys.executable, eval_script, "--compare"] + result_paths
        print(f"\nRunning eval.py comparison...")
        subprocess.run(cmd, check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
