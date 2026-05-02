#!/usr/bin/env python3
"""
Handy+ ASR benchmark evaluator.

Usage (single report):
    python eval.py benchmark/results/chinese_balanced_2026-05-01T22-00-00Z.json
    python eval.py benchmark/results/chinese_balanced_*.json benchmark/dataset/reference.txt

Usage (comparison):
    python eval.py --compare \
        benchmark/results/chinese_balanced_*.json \
        benchmark/results/multilingual_offline_*.json \
        benchmark/results/apple_native_*.json

Outputs:
    - Markdown summary printed to stdout
    - comparison_<timestamp>.md written to the same directory as the first report
    - comparison_<timestamp>.png latency histogram (requires matplotlib; skipped if absent)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Hotword recall helpers
# ---------------------------------------------------------------------------

def parse_hotwords_from_reference(lines: list[str]) -> list[str]:
    """Extract hotwords from '# hotwords: word1,word2,word3' header lines."""
    hotwords: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("# hotwords:"):
            words_part = stripped[len("# hotwords:"):].strip()
            hotwords.extend(w.strip() for w in words_part.split(",") if w.strip())
    return hotwords


def compute_hotword_recall(
    items: list[dict],
    references: list[str],
    hotwords: list[str],
) -> Optional[dict]:
    """
    For each reference line that contains a hotword, check whether the
    corresponding hypothesis also contains it.

    Returns a dict with per-hotword recall and an overall recall score,
    or None if no hotwords or no reference data.
    """
    if not hotwords or not references:
        return None

    # Filter out header lines from references
    ref_lines = [l for l in references if not l.strip().startswith("#")]

    per_word: dict[str, dict[str, int]] = {
        hw: {"total": 0, "recalled": 0} for hw in hotwords
    }

    for i, item in enumerate(items):
        if i >= len(ref_lines):
            break
        ref = ref_lines[i].strip()
        hyp = item.get("hypothesis", "")
        ref_lower = ref.lower()
        hyp_lower = hyp.lower()
        for hw in hotwords:
            hw_lower = hw.lower()
            if hw_lower in ref_lower:
                per_word[hw]["total"] += 1
                if hw_lower in hyp_lower:
                    per_word[hw]["recalled"] += 1

    total_occ = sum(v["total"] for v in per_word.values())
    total_recalled = sum(v["recalled"] for v in per_word.values())
    overall = (total_recalled / total_occ) if total_occ > 0 else None

    return {
        "overall": overall,
        "total_occurrences": total_occ,
        "total_recalled": total_recalled,
        "per_word": per_word,
    }


def format_hotword_recall_section(recall: dict) -> list[str]:
    """Render hotword recall as markdown lines."""
    lines: list[str] = []
    lines.append("## Hotword Recall")
    lines.append("")
    overall = recall["overall"]
    overall_str = f"{overall:.4f}" if overall is not None else "N/A (no occurrences)"
    lines.append(f"Overall recall: **{overall_str}**  "
                 f"({recall['total_recalled']}/{recall['total_occurrences']} occurrences)")
    lines.append("")
    lines.append("| Hotword | Occurrences | Recalled | Recall |")
    lines.append("| ------- | ----------- | -------- | ------ |")
    for hw, counts in sorted(recall["per_word"].items()):
        tot = counts["total"]
        rec = counts["recalled"]
        rate = f"{rec / tot:.4f}" if tot > 0 else "N/A"
        lines.append(f"| `{hw}` | {tot} | {rec} | {rate} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# WER helpers
# ---------------------------------------------------------------------------

def _char_edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein distance at character level (fallback when jiwer is absent)."""
    r, h = list(ref), list(hyp)
    m, n = len(r), len(h)
    # dp[i][j] = edit distance between r[:i] and h[:j]
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n]


def _normalize(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace."""
    text = text.lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFC", text)
        if not unicodedata.category(ch).startswith("P")
    )
    return " ".join(text.split())


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (edit distance at char level / ref length)."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = _char_edit_distance(ref, hyp)
    return dist / max(len(ref), 1)


def compute_wer_jiwer(reference: str, hypothesis: str) -> Optional[float]:
    """WER via jiwer if available, else None."""
    try:
        import jiwer  # type: ignore
        measures = jiwer.compute_measures(reference, hypothesis)
        return measures["wer"]
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Single-report display
# ---------------------------------------------------------------------------

def display_single(
    report: dict,
    reference_path: Optional[str] = None,
    hotword_recall: bool = False,
) -> str:
    lines: list[str] = []

    preset_id = report.get("preset_id", "?")
    preset_name = report.get("preset_name", preset_id)
    model_id = report.get("model_id", "?")
    language = report.get("language", "?")
    punc_zh = report.get("punc_zh_enabled", False)
    timestamp = report.get("timestamp", "")
    summary = report.get("summary", {})
    items = report.get("items", [])

    lines.append(f"# Benchmark Report: {preset_name}")
    lines.append("")
    lines.append(f"| Field             | Value                     |")
    lines.append(f"| ----------------- | ------------------------- |")
    lines.append(f"| Preset ID         | `{preset_id}`             |")
    lines.append(f"| Model             | `{model_id}`              |")
    lines.append(f"| Language          | `{language}`              |")
    lines.append(f"| Punc-ZH enabled   | {punc_zh}                 |")
    lines.append(f"| Timestamp         | {timestamp}               |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric                | Value                  |")
    lines.append(f"| --------------------- | ---------------------- |")
    lines.append(f"| Total items           | {summary.get('total_items', 0)} |")
    lines.append(f"| Total audio (s)       | {summary.get('total_audio_seconds', 0):.1f} |")
    lines.append(f"| P50 latency (ms)      | {summary.get('p50_latency_ms', 0)} |")
    lines.append(f"| P95 latency (ms)      | {summary.get('p95_latency_ms', 0)} |")
    lines.append(f"| Punctuation density   | {summary.get('punctuation_density', 0):.4f} |")
    lines.append("")

    # WER/CER against reference if provided
    references: list[str] = []
    if reference_path:
        ref_file = Path(reference_path)
        if ref_file.exists():
            references = ref_file.read_text(encoding="utf-8").splitlines()
        else:
            lines.append(f"> WARNING: reference file not found: {reference_path}")

    if references and items:
        # Filter out header lines for WER/CER computation
        ref_lines_plain = [l for l in references if not l.strip().startswith("#")]
        total_cer = 0.0
        total_wer_j: Optional[float] = 0.0
        counted = 0
        for i, item in enumerate(items):
            if i >= len(ref_lines_plain) or not ref_lines_plain[i].strip():
                continue
            ref = ref_lines_plain[i].strip()
            hyp = item.get("hypothesis", "")
            total_cer += compute_cer(ref, hyp)
            wj = compute_wer_jiwer(ref, hyp)
            if wj is not None and total_wer_j is not None:
                total_wer_j += wj
            else:
                total_wer_j = None
            counted += 1

        if counted > 0:
            avg_cer = total_cer / counted
            lines.append("## Error Rates")
            lines.append("")
            lines.append(f"| Metric | Value  | Note |")
            lines.append(f"| ------ | ------ | ---- |")
            lines.append(f"| CER    | {avg_cer:.4f} | char-level (custom) |")
            if total_wer_j is not None:
                avg_wer = total_wer_j / counted
                lines.append(f"| WER    | {avg_wer:.4f} | jiwer               |")
            else:
                lines.append(f"| WER    | N/A    | install jiwer for WER |")
            lines.append(f"| Files evaluated | {counted}/{len(items)} | |")
            lines.append("")

    # Hotword recall (only when --hotword-recall flag is set and reference provided)
    if hotword_recall and references and items:
        detected_hotwords = parse_hotwords_from_reference(references)
        if detected_hotwords:
            recall = compute_hotword_recall(items, references, detected_hotwords)
            if recall is not None:
                lines.extend(format_hotword_recall_section(recall))
        else:
            lines.append("> NOTE: --hotword-recall enabled but no `# hotwords:` header found in reference.")
            lines.append("")

    # Sample hypotheses
    lines.append("## Sample Hypotheses (first 5)")
    lines.append("")
    for item in items[:5]:
        wav = item.get("wav", "?")
        hyp = item.get("hypothesis", "")
        lat = item.get("latency_ms", 0)
        lines.append(f"- **{wav}** ({lat} ms): {hyp or '*(empty)*'}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def display_comparison(reports: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# ASR Preset Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("")

    headers = ["Metric"] + [r.get("preset_name", r.get("preset_id", "?")) for r in reports]
    rows: list[list[str]] = []

    def _get(r: dict, key: str, fmt: str = "{}") -> str:
        val = r.get("summary", {}).get(key, "N/A")
        if val == "N/A":
            return "N/A"
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    rows.append(["Model"] + [r.get("model_id", "?") for r in reports])
    rows.append(["Language"] + [r.get("language", "?") for r in reports])
    rows.append(["Punc-ZH"] + [str(r.get("punc_zh_enabled", False)) for r in reports])
    rows.append(["Total items"] + [_get(r, "total_items") for r in reports])
    rows.append(["Total audio (s)"] + [_get(r, "total_audio_seconds", "{:.1f}") for r in reports])
    rows.append(["P50 latency (ms)"] + [_get(r, "p50_latency_ms") for r in reports])
    rows.append(["P95 latency (ms)"] + [_get(r, "p95_latency_ms") for r in reports])
    rows.append(["Punc density"] + [_get(r, "punctuation_density", "{:.4f}") for r in reports])

    # Compute column widths
    all_rows = [headers] + rows
    col_widths = [max(len(str(row[c])) for row in all_rows) for c in range(len(headers))]

    def _fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    def _sep_row() -> str:
        return "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    lines.append(_fmt_row(headers))
    lines.append(_sep_row())
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Latency histogram
# ---------------------------------------------------------------------------

def plot_latency_histogram(reports: list[dict], output_path: str) -> bool:
    """Draw a latency histogram for each report. Returns True if saved."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue", "tomato", "seagreen", "goldenrod", "mediumpurple"]

    for i, report in enumerate(reports):
        latencies = [item.get("latency_ms", 0) for item in report.get("items", [])]
        if not latencies:
            continue
        label = report.get("preset_name", report.get("preset_id", f"preset_{i}"))
        ax.hist(latencies, bins=20, alpha=0.6, label=label, color=colors[i % len(colors)])

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("ASR Latency Distribution by Preset")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Handy+ ASR benchmark evaluator")
    parser.add_argument(
        "reports",
        nargs="+",
        help="Path(s) to JSON report file(s). In --compare mode, pass all reports here.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple reports side by side.",
    )
    parser.add_argument(
        "--reference",
        metavar="FILE",
        default=None,
        help="Path to reference.txt (one line per WAV, sorted by filename). Used for CER/WER.",
    )
    parser.add_argument(
        "--hotword-recall",
        action="store_true",
        help=(
            "Compute hotword recall metric. Requires --reference; reference file may contain "
            "'# hotwords: word1,word2,word3' header line(s) listing words to track."
        ),
    )
    args = parser.parse_args()

    # Expand any glob patterns (useful when the shell doesn't expand them, e.g. Windows)
    import glob as _glob
    expanded: list[str] = []
    for pattern in args.reports:
        matched = _glob.glob(pattern)
        if matched:
            expanded.extend(sorted(matched))
        else:
            expanded.append(pattern)

    if not expanded:
        print("ERROR: no report files found.", file=sys.stderr)
        sys.exit(1)

    loaded = []
    for path in expanded:
        try:
            loaded.append(load_report(path))
        except Exception as exc:
            print(f"WARNING: could not load {path}: {exc}", file=sys.stderr)

    if not loaded:
        print("ERROR: no valid report files loaded.", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    first_report_dir = Path(expanded[0]).parent

    if args.compare or len(loaded) > 1:
        md = display_comparison(loaded)
        print(md)

        md_path = first_report_dir / f"comparison_{timestamp}.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"\nMarkdown report saved to: {md_path}", file=sys.stderr)

        png_path = str(first_report_dir / f"comparison_{timestamp}.png")
        if plot_latency_histogram(loaded, png_path):
            print(f"Latency histogram saved to: {png_path}", file=sys.stderr)
        else:
            print(
                "NOTE: matplotlib not found; latency histogram skipped. "
                "Install with: pip install matplotlib",
                file=sys.stderr,
            )
    else:
        md = display_single(
            loaded[0],
            reference_path=args.reference,
            hotword_recall=args.hotword_recall,
        )
        print(md)

        md_path = first_report_dir / f"{loaded[0].get('preset_id', 'report')}_{timestamp}.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"\nMarkdown report saved to: {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
