#!/usr/bin/env python3
"""
v2t-logs — CLI tool for analysing v2t observability JSONL logs.

Usage:
  v2t-logs tail [--last N] [--field K]
  v2t-logs trace <req_id>
  v2t-logs slowest [--stage S] [--top N] [--since Xd]
  v2t-logs percentiles --stage S [--preset P] [--metric M] [--since Xd]
  v2t-logs breakdown [--since Xd]
  v2t-logs compare --baseline FILE --current FILE [--stage S]
  v2t-logs assert --stage S [--metric M] [--preset P] --p50-max N --p99-max M

Exit codes: 0 = pass/ok, 1 = assert failure / error
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
from datetime import datetime, timedelta, timezone

# ── Log file discovery ────────────────────────────────────────────────────────

DEFAULT_LOG_DIR = pathlib.Path.home() / "Library" / "Logs" / "v2t"


def find_log_files(log_dir: pathlib.Path = DEFAULT_LOG_DIR):
    """Return all .jsonl files sorted oldest→newest."""
    if not log_dir.exists():
        return []
    files = sorted(log_dir.glob("v2t.jsonl*"))
    return files


def iter_events(log_files, since: datetime | None = None):
    """Yield parsed JSON objects from all log files, newest last."""
    for f in log_files:
        try:
            opener = open
            if str(f).endswith(".gz"):
                import gzip
                opener = gzip.open
            with opener(f, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if since is not None:
                        ts = obj.get("fields", {}).get("ts_unix_ms") or obj.get("ts_unix_ms")
                        if ts is not None:
                            event_dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
                            if event_dt < since:
                                continue
                    yield obj
        except (OSError, IOError):
            continue


def parse_since(since_str: str | None) -> datetime | None:
    """Parse e.g. '7d', '24h', '30m' into a datetime cutoff."""
    if not since_str:
        return None
    s = since_str.strip()
    if s.endswith("d"):
        return datetime.now(tz=timezone.utc) - timedelta(days=int(s[:-1]))
    if s.endswith("h"):
        return datetime.now(tz=timezone.utc) - timedelta(hours=int(s[:-1]))
    if s.endswith("m"):
        return datetime.now(tz=timezone.utc) - timedelta(minutes=int(s[:-1]))
    return None


def extract_stage_events(events):
    """Filter to v2t stage events only."""
    for e in events:
        fields = e.get("fields", e)  # tracing-subscriber json wraps in "fields"
        if not isinstance(fields, dict):
            continue
        stage = fields.get("stage")
        if stage and fields.get("request_id"):
            yield fields


def get_metric(event: dict, metric: str | None) -> float | None:
    """Extract a numeric metric from an event's extra field or duration_ms."""
    if metric is None or metric == "duration_ms":
        v = event.get("duration_ms")
        return float(v) if v is not None else None

    # Try to parse from the 'extra' string field: "key=value key2=value2"
    extra_str = event.get("extra", "")
    for part in extra_str.split():
        if "=" in part:
            k, v = part.split("=", 1)
            if k == metric:
                try:
                    return float(v)
                except ValueError:
                    return None

    # Fallback: check for the metric as a top-level key
    v = event.get(metric)
    if v is not None:
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    return None


def percentile(data: list[float], p: float) -> float:
    """Compute percentile p (0–100) via linear interpolation."""
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_tail(args):
    files = find_log_files()
    events = list(iter_events(files))
    last = getattr(args, "last", 20) or 20
    recent = events[-last:]
    field = getattr(args, "field", None)
    for e in recent:
        fields = e.get("fields", e)
        if field:
            print(fields.get(field, ""))
        else:
            print(json.dumps(fields, ensure_ascii=False))


def cmd_trace(args):
    req_id = args.req_id
    files = find_log_files()
    stage_order = [
        "t0_hotkey", "t1_audio_capture", "t2_recording",
        "t3_vad", "t4_resample", "t5_inference",
        "t5a_chunk_inference", "t5b_final_pass",
        "t6_postprocess", "t7_output", "total",
    ]
    found: dict[str, dict] = {}
    for e in extract_stage_events(iter_events(files)):
        if e.get("request_id") == req_id:
            stage = e.get("stage", "")
            found[stage] = e

    if not found:
        print(f"No events found for request_id={req_id}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWaterfall for request {req_id}\n{'─'*60}")
    max_stage_len = max(len(s) for s in stage_order)
    for stage in stage_order:
        e = found.get(stage)
        if e:
            dur = e.get("duration_ms", "?")
            outcome = e.get("outcome", "?")
            extra = e.get("extra", "")
            dur_str = f"{float(dur):.1f}ms" if dur != "?" else "?"
            print(f"  {stage:<{max_stage_len}}  {outcome:<10}  {dur_str:<10}  {extra}")
        else:
            print(f"  {stage:<{max_stage_len}}  {'(missing)'}")
    print()


def cmd_slowest(args):
    since = parse_since(getattr(args, "since", None))
    stage_filter = getattr(args, "stage", None)
    top = getattr(args, "top", 10) or 10
    files = find_log_files()
    events = list(extract_stage_events(iter_events(files, since=since)))
    if stage_filter:
        events = [e for e in events if e.get("stage") == stage_filter]
    # Sort by duration_ms descending
    events.sort(key=lambda e: float(e.get("duration_ms", 0)), reverse=True)
    header = f"{'stage':<20} {'request_id':<27} {'duration_ms':>12} {'outcome':<10} extra"
    print(header)
    print("─" * 90)
    for e in events[:top]:
        stage = e.get("stage", "")
        req = e.get("request_id", "")
        dur = e.get("duration_ms", 0)
        outcome = e.get("outcome", "")
        extra = e.get("extra", "")[:40]
        print(f"  {stage:<18} {req:<27} {float(dur):>12.1f} {outcome:<10} {extra}")


def cmd_percentiles(args):
    stage = args.stage
    metric = getattr(args, "metric", None)
    preset = getattr(args, "preset", None)
    since = parse_since(getattr(args, "since", None))
    files = find_log_files()
    data: list[float] = []
    for e in extract_stage_events(iter_events(files, since=since)):
        if e.get("stage") != stage:
            continue
        if preset:
            extra = e.get("extra", "")
            if f"preset={preset}" not in extra:
                continue
        v = get_metric(e, metric)
        if v is not None and not math.isnan(v):
            data.append(v)
    if not data:
        print(f"No data for stage={stage} metric={metric} preset={preset}")
        return
    metric_label = metric or "duration_ms"
    print(f"\nPercentiles: stage={stage}  metric={metric_label}  n={len(data)}\n{'─'*40}")
    for p in [50, 75, 90, 95, 99, 100]:
        v = percentile(data, p)
        print(f"  p{p:<3}  {v:>10.2f}")
    print(f"  mean {statistics.mean(data):>10.2f}")
    print()


def cmd_breakdown(args):
    since = parse_since(getattr(args, "since", None))
    files = find_log_files()
    stage_totals: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    total_end_to_end: list[float] = []
    for e in extract_stage_events(iter_events(files, since=since)):
        stage = e.get("stage", "")
        dur = e.get("duration_ms")
        if dur is None:
            continue
        try:
            dur_f = float(dur)
        except (ValueError, TypeError):
            continue
        stage_totals[stage] = stage_totals.get(stage, 0.0) + dur_f
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if stage == "total":
            total_end_to_end.append(dur_f)

    total_sum = sum(v for k, v in stage_totals.items() if k != "total")
    if total_sum == 0:
        print("No stage data found.")
        return

    stage_order = [
        "t0_hotkey", "t1_audio_capture", "t2_recording",
        "t3_vad", "t4_resample", "t5_inference",
        "t5a_chunk_inference", "t5b_final_pass",
        "t6_postprocess", "t7_output",
    ]
    print(f"\nStage Breakdown  (n_total={stage_counts.get('total', 0)})\n{'─'*60}")
    print(f"  {'stage':<20} {'n':>6} {'mean_ms':>10} {'pct_total':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*10}")
    for s in stage_order:
        tot = stage_totals.get(s, 0.0)
        cnt = stage_counts.get(s, 0)
        mean = tot / cnt if cnt > 0 else 0.0
        pct = (tot / total_sum * 100) if total_sum > 0 else 0.0
        print(f"  {s:<20} {cnt:>6} {mean:>10.1f} {pct:>9.1f}%")
    if total_end_to_end:
        p50 = percentile(total_end_to_end, 50)
        p99 = percentile(total_end_to_end, 99)
        print(f"\n  End-to-end p50={p50:.0f}ms  p99={p99:.0f}ms  n={len(total_end_to_end)}")
    print()


def load_stage_data(log_file: pathlib.Path, stage_filter: str | None = None) -> list[dict]:
    """Load events from a single file."""
    results = []
    try:
        with open(log_file, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fields = obj.get("fields", obj)
                if not isinstance(fields, dict):
                    continue
                if not fields.get("stage"):
                    continue
                if stage_filter and fields.get("stage") != stage_filter:
                    continue
                results.append(fields)
    except (OSError, IOError):
        pass
    return results


def cmd_compare(args):
    baseline_file = pathlib.Path(args.baseline)
    current_file = pathlib.Path(args.current)
    stage = getattr(args, "stage", None)

    baseline_events = load_stage_data(baseline_file, stage)
    current_events = load_stage_data(current_file, stage)

    def group_by_stage(events):
        groups: dict[str, list[float]] = {}
        for e in events:
            s = e.get("stage", "")
            dur = e.get("duration_ms")
            if dur is not None:
                groups.setdefault(s, []).append(float(dur))
        return groups

    bl = group_by_stage(baseline_events)
    cur = group_by_stage(current_events)
    all_stages = sorted(set(list(bl.keys()) + list(cur.keys())))

    print(f"\nComparison: {baseline_file.name} → {current_file.name}")
    if stage:
        print(f"  (filtered to stage={stage})")
    print(f"  {'stage':<20} {'bl_p50':>8} {'cur_p50':>8} {'delta':>8} {'bl_p99':>8} {'cur_p99':>8} {'delta':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for s in all_stages:
        bl_data = bl.get(s, [])
        cur_data = cur.get(s, [])
        bl_p50 = percentile(bl_data, 50) if bl_data else float("nan")
        cur_p50 = percentile(cur_data, 50) if cur_data else float("nan")
        bl_p99 = percentile(bl_data, 99) if bl_data else float("nan")
        cur_p99 = percentile(cur_data, 99) if cur_data else float("nan")
        d50 = cur_p50 - bl_p50 if not (math.isnan(cur_p50) or math.isnan(bl_p50)) else float("nan")
        d99 = cur_p99 - bl_p99 if not (math.isnan(cur_p99) or math.isnan(bl_p99)) else float("nan")
        sign50 = "+" if d50 > 0 else ""
        sign99 = "+" if d99 > 0 else ""
        bl_p50_s = f"{bl_p50:.1f}" if not math.isnan(bl_p50) else "n/a"
        cur_p50_s = f"{cur_p50:.1f}" if not math.isnan(cur_p50) else "n/a"
        d50_s = f"{sign50}{d50:.1f}" if not math.isnan(d50) else "n/a"
        bl_p99_s = f"{bl_p99:.1f}" if not math.isnan(bl_p99) else "n/a"
        cur_p99_s = f"{cur_p99:.1f}" if not math.isnan(cur_p99) else "n/a"
        d99_s = f"{sign99}{d99:.1f}" if not math.isnan(d99) else "n/a"
        print(f"  {s:<20} {bl_p50_s:>8} {cur_p50_s:>8} {d50_s:>8} {bl_p99_s:>8} {cur_p99_s:>8} {d99_s:>8}")
    print()


def cmd_export_baseline(args):
    """Export a percentile snapshot for all stages as a baseline JSON file.

    Usage:
        v2t-logs export-baseline --preset P --version V --since 7d
    Output JSON format:
        {
          "preset": "...",
          "version": "...",
          "snapshot_date": "2026-...",
          "stages": {
            "t5_inference": {"p50_ms": 120.0, "p95_ms": 210.0, "p99_ms": 280.0, "n": 42}
          }
        }
    If no data is found, reports an error and exits 1 without writing.
    """
    preset = getattr(args, "preset", None)
    version = getattr(args, "version", "unknown")
    since = parse_since(getattr(args, "since", "7d"))
    metric = getattr(args, "metric", None)

    files = find_log_files()
    stage_data: dict[str, list[float]] = {}

    for e in extract_stage_events(iter_events(files, since=since)):
        if preset:
            extra = e.get("extra", "")
            if f"preset={preset}" not in extra and e.get("stage") not in (
                "t0_hotkey", "t1_audio_capture", "t2_recording", "t4_resample",
                "t6_postprocess", "t7_output", "total",
            ):
                # For non-preset-specific stages, keep all events
                pass
            elif preset and f"preset={preset}" not in extra and e.get("stage") == "t5_inference":
                continue  # skip inference events for other presets
        stage = e.get("stage", "")
        if not stage:
            continue
        v = get_metric(e, metric)
        if v is not None and not math.isnan(v):
            stage_data.setdefault(stage, []).append(v)

    if not stage_data:
        print(
            f"No data for preset={preset} since={getattr(args, 'since', '7d')}. "
            "Would write empty stages. Run after >= 7 days of real usage.",
            file=sys.stderr,
        )
        sys.exit(1)

    stages_out: dict[str, dict] = {}
    for stage, vals in sorted(stage_data.items()):
        stages_out[stage] = {
            "p50_ms": round(percentile(vals, 50), 2),
            "p95_ms": round(percentile(vals, 95), 2),
            "p99_ms": round(percentile(vals, 99), 2),
            "n": len(vals),
        }

    result = {
        "preset": preset,
        "version": version,
        "snapshot_date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stages": stages_out,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_assert(args):
    stage = args.stage
    metric = getattr(args, "metric", None)
    preset = getattr(args, "preset", None)
    p50_max = getattr(args, "p50_max", None)
    p99_max = getattr(args, "p99_max", None)
    since = parse_since(getattr(args, "since", "7d"))

    files = find_log_files()
    data: list[float] = []
    for e in extract_stage_events(iter_events(files, since=since)):
        if e.get("stage") != stage:
            continue
        if preset:
            extra = e.get("extra", "")
            if f"preset={preset}" not in extra:
                continue
        v = get_metric(e, metric)
        if v is not None and not math.isnan(v):
            data.append(v)

    metric_label = metric or "duration_ms"

    if not data:
        print(
            f"SKIP  stage={stage} metric={metric_label} preset={preset}: no data (0 samples)",
            file=sys.stderr,
        )
        # No data → skip (do not fail gate; data is produced by manual testing)
        sys.exit(0)

    p50 = percentile(data, 50)
    p99 = percentile(data, 99)
    passed = True
    msgs = []

    if p50_max is not None and p50 > float(p50_max):
        msgs.append(f"p50={p50:.2f} > max={p50_max}")
        passed = False
    if p99_max is not None and p99 > float(p99_max):
        msgs.append(f"p99={p99:.2f} > max={p99_max}")
        passed = False

    label = f"stage={stage} metric={metric_label} preset={preset} n={len(data)}"
    if passed:
        print(f"PASS  {label}  p50={p50:.2f}  p99={p99:.2f}")
        sys.exit(0)
    else:
        print(f"FAIL  {label}  p50={p50:.2f}  p99={p99:.2f}  violations: {', '.join(msgs)}")
        sys.exit(1)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="v2t-logs",
        description="v2t observability log analyser",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # tail
    p_tail = sub.add_parser("tail", help="Print most recent log entries")
    p_tail.add_argument("--last", type=int, default=20, metavar="N")
    p_tail.add_argument("--field", type=str, default=None, metavar="K")

    # trace
    p_trace = sub.add_parser("trace", help="Waterfall view for a single request")
    p_trace.add_argument("req_id", help="ULID request id")

    # slowest
    p_slow = sub.add_parser("slowest", help="Slowest N pipeline runs")
    p_slow.add_argument("--stage", type=str, default=None)
    p_slow.add_argument("--top", type=int, default=10)
    p_slow.add_argument("--since", type=str, default=None, metavar="Xd")

    # percentiles
    p_pct = sub.add_parser("percentiles", help="Latency percentiles for a stage")
    p_pct.add_argument("--stage", type=str, required=True)
    p_pct.add_argument("--preset", type=str, default=None)
    p_pct.add_argument("--metric", type=str, default=None)
    p_pct.add_argument("--since", type=str, default=None, metavar="Xd")

    # breakdown
    p_bd = sub.add_parser("breakdown", help="Stage pct of total pipeline time")
    p_bd.add_argument("--since", type=str, default=None, metavar="Xd")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two JSONL files")
    p_cmp.add_argument("--baseline", required=True, metavar="FILE")
    p_cmp.add_argument("--current", required=True, metavar="FILE")
    p_cmp.add_argument("--stage", type=str, default=None)

    # assert
    p_assert = sub.add_parser("assert", help="Hard gate: exit 1 if SLA violated")
    p_assert.add_argument("--stage", type=str, required=True)
    p_assert.add_argument("--metric", type=str, default=None)
    p_assert.add_argument("--preset", type=str, default=None)
    p_assert.add_argument("--p50-max", type=float, default=None, dest="p50_max")
    p_assert.add_argument("--p99-max", type=float, default=None, dest="p99_max")
    p_assert.add_argument("--since", type=str, default="7d", metavar="Xd")

    # export-baseline
    p_eb = sub.add_parser(
        "export-baseline",
        help="Export percentile snapshot for all stages as baseline JSON (stdout)",
    )
    p_eb.add_argument("--preset", type=str, default=None, help="Filter to a specific preset")
    p_eb.add_argument("--version", type=str, default="unknown", help="Version label (e.g. 1.0.0)")
    p_eb.add_argument("--since", type=str, default="7d", metavar="Xd",
                      help="Time window (e.g. 7d, 24h). Default: 7d")
    p_eb.add_argument("--metric", type=str, default=None,
                      help="Metric field to aggregate (default: duration_ms)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "tail": cmd_tail,
        "trace": cmd_trace,
        "slowest": cmd_slowest,
        "percentiles": cmd_percentiles,
        "breakdown": cmd_breakdown,
        "compare": cmd_compare,
        "assert": cmd_assert,
        "export-baseline": cmd_export_baseline,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
