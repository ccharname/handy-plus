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


class CtPuncEngine:
    """CT-Transformer Chinese punctuation via sherpa-onnx OfflinePunctuation.

    Mirrors the Rust apply_punc_zh_if_applicable logic:
    - Only applied when punc_zh_enabled=True
    - Applied whenever text contains CJK characters (content-based fallback)
    - Returns text unchanged if model is absent or inference fails
    """

    def __init__(self, models_base: str):
        import sherpa_onnx

        model_dir = os.path.join(
            models_base,
            "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
        )
        onnx_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"CT-Punc model not found: {onnx_path}")

        config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
                ct_transformer=onnx_path,
                num_threads=1,
            )
        )
        self.punct = sherpa_onnx.OfflinePunctuation(config)

    @staticmethod
    def _text_has_cjk(text: str) -> bool:
        return any(
            0x3400 <= ord(c) <= 0x4DBF  # CJK Ext A
            or 0x4E00 <= ord(c) <= 0x9FFF  # CJK Unified
            or 0xF900 <= ord(c) <= 0xFAFF  # CJK Compatibility
            or 0x3000 <= ord(c) <= 0x303F  # CJK Symbols & Punctuation
            for c in text
        )

    def apply(self, text: str, lang: str = "auto") -> str:
        """Apply punctuation if conditions match (mirrors Rust logic)."""
        if not text:
            return text
        lang_says_zh = lang.split("-")[0] == "zh" or lang == "yue"
        if not lang_says_zh and not self._text_has_cjk(text):
            return text
        try:
            result = self.punct.add_punctuation(text)
            return result if result else text
        except Exception:
            return text


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

    # Load CT-Punc engine for presets that have punc_zh_enabled=True.
    # This mirrors Rust apply_punc_zh_if_applicable which is always in the pipeline.
    punc_engine: Optional[CtPuncEngine] = None
    if meta.get("punc_zh_enabled"):
        try:
            print("Loading CT-Punc model... ", end="", flush=True)
            t_punc_load = time.monotonic()
            punc_engine = CtPuncEngine(models_base)
            print(f"ready ({int((time.monotonic() - t_punc_load) * 1000)} ms)")
        except FileNotFoundError as exc:
            print(f"SKIP (not found: {exc})")
        except Exception as exc:
            print(f"SKIP (error: {exc})")

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

        # Apply CT-Punc post-processing (mirrors Rust pipeline).
        if punc_engine is not None and text and not text.startswith("["):
            text = punc_engine.apply(text, lang=meta.get("language", "auto"))

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

def _summary_metrics(report: dict) -> dict:
    """Extract the three metrics the regression gate cares about."""
    summary = report.get("summary", {}) or {}
    return {
        "p50": summary.get("steady_p50_latency_ms") or summary.get("p50_latency_ms"),
        "p95": summary.get("steady_p95_latency_ms") or summary.get("p95_latency_ms"),
        "punc_density": summary.get("punctuation_density"),
        "items": summary.get("total_items"),
    }


def _newest_baseline_for(preset_id: str, baseline_path: str) -> Optional[str]:
    """Resolve --baseline arg to a concrete JSON file path for a given preset.

    Accepts either a directory (search for newest matching `{preset_id}_*.json`,
    skipping derivative files like `*_punc_only_*` / `*_swap_*`) or a single
    JSON file (returned unchanged when its name matches the preset).
    """
    p = Path(baseline_path)
    if p.is_file():
        if p.name.startswith(f"{preset_id}_"):
            return str(p)
        return None
    if not p.is_dir():
        return None
    candidates = sorted(
        (
            f for f in p.rglob(f"{preset_id}_*.json")
            if "_punc_only_" not in f.name and "_swap_" not in f.name
        ),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def check_regression(
    preset_id: str,
    new_report: dict,
    baseline_path: str,
    threshold_pct: float,
) -> tuple[bool, str]:
    """Compare new bench results against a baseline JSON.

    Returns `(is_regression, message)`. `is_regression=True` when p50 or p95
    worsens by more than `threshold_pct` percent or when error_count grows.
    """
    base_file = _newest_baseline_for(preset_id, baseline_path)
    if base_file is None:
        return False, f"[{preset_id}] no baseline file found for preset"
    try:
        with open(base_file, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"[{preset_id}] failed to read baseline {base_file}: {exc}"

    base = _summary_metrics(baseline)
    new = _summary_metrics(new_report)

    msgs = []
    is_regression = False
    for metric in ("p50", "p95"):
        b, n = base.get(metric), new.get(metric)
        if not b or not n:
            continue
        delta_pct = (n - b) / b * 100.0
        marker = ""
        if delta_pct > threshold_pct:
            is_regression = True
            marker = " ❌ REGRESSION"
        elif delta_pct < -threshold_pct:
            marker = " ✅ improved"
        msgs.append(f"{metric}: {b:>5} → {n:>5} ms ({delta_pct:+.1f}%){marker}")

    base_errors = (baseline.get("summary", {}) or {}).get("errors")
    new_errors = (new_report.get("summary", {}) or {}).get("errors")
    if base_errors is not None and new_errors is not None and new_errors > base_errors:
        is_regression = True
        msgs.append(f"errors: {base_errors} → {new_errors} ❌ REGRESSION")

    header = f"[{preset_id}] baseline={Path(base_file).name}"
    return is_regression, "\n  ".join([header] + msgs)


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
    parser.add_argument("--baseline", default="",
                        help="Path to a baseline JSON or directory to compare against. "
                             "If a directory is given the newest matching report per preset is picked. "
                             "When set, exit code 2 is returned if any preset regresses by >--regression-pct.")
    parser.add_argument("--regression-pct", type=float, default=15.0,
                        help="Regression threshold (percent worse than baseline) for p50 / p95 (default: 15)")
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
    new_reports: dict[str, dict] = {}
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
            new_reports[preset_id] = result["report"]
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

    # Regression gate (X1) — runs after eval comparison so the user sees both views.
    if args.baseline:
        print(f"\n{'='*60}")
        print(f"Regression gate vs baseline: {args.baseline} (threshold {args.regression_pct}%)")
        any_regression = False
        for preset_id, report in new_reports.items():
            is_regression, msg = check_regression(
                preset_id, report, args.baseline, args.regression_pct
            )
            print(f"  {msg}")
            any_regression = any_regression or is_regression
        if any_regression:
            print("\nRESULT: at least one preset regressed past the threshold.")
            return 2
        print("\nRESULT: all presets within threshold.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
