#!/usr/bin/env python3
"""
Handy+ benchmark runner — triggers run_asr_benchmark in the running Handy.app
via the tauri-plugin-single-instance Unix socket, then waits for result JSON.

Usage:
    python benchmark/run_all.py [--presets <id,...>] [--dataset <dir>] [--output <dir>]

Defaults:
    presets  = chinese_balanced,multilingual_offline,apple_native
    dataset  = benchmark/dataset
    output   = benchmark/results

How it works:
  1. Connects to /tmp/com_pais_handy_si.sock (Tauri single-instance IPC).
  2. Sends: {cwd}\\0\\0handy\\0--bench-preset\\0<id>\\0--bench-dataset\\0<dir>\\0--bench-output\\0<dir>
  3. The running Handy app's single-instance callback picks up the args and
     calls run_asr_benchmark(), writing a JSON report to output_dir.
  4. This script polls output_dir for a new *.json file matching the preset,
     waiting up to TIMEOUT_SECS.
  5. After all presets finish, runs eval.py --compare on the collected reports.
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SOCKET_PATH = "/tmp/com_pais_handy_si.sock"
TIMEOUT_SECS = 600  # 10 min per preset (large model load + 38 files)
POLL_INTERVAL = 2.0
DEFAULT_PRESETS = ["chinese_balanced", "multilingual_offline", "apple_native"]


def send_bench_trigger(preset_id: str, dataset_dir: str, output_dir: str) -> None:
    """Send bench args to the running Handy via single-instance socket."""
    cwd = os.getcwd()
    # Protocol: {cwd}\0\0{arg0}\0{arg1}\0...
    args = ["handy", "--bench-preset", preset_id, "--bench-dataset", dataset_dir, "--bench-output", output_dir]
    payload = cwd + "\0\0" + "\0".join(args)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(SOCKET_PATH)
        sock.sendall(payload.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
    finally:
        sock.close()


def wait_for_result(preset_id: str, output_dir: str, before_files: set, timeout: float) -> str | None:
    """
    Poll output_dir for a new JSON file whose name starts with preset_id.
    Returns the path if found within timeout, else None.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = set(glob.glob(os.path.join(output_dir, f"{preset_id}_*.json")))
        new = candidates - before_files
        if new:
            return sorted(new)[-1]  # latest
        time.sleep(POLL_INTERVAL)
    return None


def run_preset(preset_id: str, dataset_dir: str, output_dir: str) -> str | None:
    """Trigger benchmark for one preset and wait for result. Returns JSON path or None."""
    print(f"\n[{preset_id}] Triggering benchmark...")

    # Snapshot existing JSON files before triggering
    before = set(glob.glob(os.path.join(output_dir, f"{preset_id}_*.json")))

    try:
        send_bench_trigger(preset_id, dataset_dir, output_dir)
    except Exception as exc:
        print(f"  ERROR sending trigger: {exc}", file=sys.stderr)
        return None

    print(f"  Trigger sent. Waiting up to {TIMEOUT_SECS}s for result JSON...")
    result_path = wait_for_result(preset_id, output_dir, before, TIMEOUT_SECS)
    if result_path:
        print(f"  Result: {result_path}")
    else:
        print(f"  TIMEOUT: no result JSON for {preset_id} after {TIMEOUT_SECS}s", file=sys.stderr)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Handy+ benchmark runner")
    parser.add_argument("--presets", default=",".join(DEFAULT_PRESETS),
                        help="Comma-separated preset IDs (default: all three)")
    parser.add_argument("--dataset", default="benchmark/dataset",
                        help="Dataset directory with WAV files")
    parser.add_argument("--output", default="benchmark/results",
                        help="Output directory for JSON reports")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip running eval.py after benchmarks")
    args = parser.parse_args()

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    dataset_dir = str(Path(args.dataset).resolve())
    output_dir = str(Path(args.output).resolve())

    # Ensure paths exist
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check socket
    if not os.path.exists(SOCKET_PATH):
        print(f"ERROR: Handy single-instance socket not found: {SOCKET_PATH}", file=sys.stderr)
        print("Is Handy.app running? Launch it first.", file=sys.stderr)
        return 1

    # Check dataset
    wav_files = list(Path(dataset_dir).glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files found in {dataset_dir}", file=sys.stderr)
        return 1
    print(f"Dataset: {len(wav_files)} WAV files in {dataset_dir}")

    result_paths = []
    for preset_id in presets:
        path = run_preset(preset_id, dataset_dir, output_dir)
        if path:
            result_paths.append(path)
        else:
            print(f"WARNING: {preset_id} benchmark did not produce a result.", file=sys.stderr)

    if not result_paths:
        print("\nERROR: No benchmark results produced.", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"Benchmarks complete. {len(result_paths)}/{len(presets)} presets succeeded.")
    for p in result_paths:
        print(f"  {p}")

    if not args.skip_eval and len(result_paths) >= 1:
        eval_script = str(Path(__file__).parent / "eval.py")
        cmd = [sys.executable, eval_script, "--compare"] + result_paths
        print(f"\nRunning eval.py comparison...")
        subprocess.run(cmd, check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
