#!/usr/bin/env bash
# FD-009 7-day audit reproducer. Run on day 7 (anchor commit b344c57+TTL fix
# + CJK filler fix, deployed 2026-05-07 ~10:15) to compare 7 days of real
# usage against the deploy-day baseline that shipped with that commit.
#
# What it produces:
#   1. Stability metrics — process uptime, RSS, CPU, ERROR/panic count
#   2. M2/M3/M4 telemetry — actual hit rates over the week
#   3. CJK collapse / filler regression scan — residual patterns in transcripts
#   4. A/B sample re-run — pick 5 wavs from the past week, compare
#      no-biasing baseline (test_mlx) vs current pipeline output
#   5. Aggregate report → /tmp/v2t-audit-7d-<date>.md
#
# Anchor commit: b344c57 (CJK filler) — search for telemetry log lines
# that ship with that release: "Tap detected" / "Hold detected" /
# "lexical biasing context" / "context biasing with N recent transcripts".

set -euo pipefail

LOG="$HOME/Library/Logs/dev.zheng.v2t/v2t.log"
ROT="$LOG.1"  # 50 MB rotation backup, may not exist
DB="$HOME/Library/Application Support/dev.zheng.v2t/history.db"
RECDIR="$HOME/Library/Application Support/dev.zheng.v2t/recordings"
EXAMPLE="/Users/zhengma/Developer/handy/src-tauri/target/release/examples/test_mlx"
OUT="/tmp/v2t-audit-7d-$(date +%Y%m%d-%H%M).md"
ANCHOR_TS=1778117100   # 2026-05-07 ~09:21 (CJK collapse deploy commit c40cc33)

# Concatenate active log + rotation backup so we span the full 7 days.
LOGCAT=$(mktemp)
trap "rm -f $LOGCAT" EXIT
[ -f "$ROT" ] && cat "$ROT" > "$LOGCAT"
cat "$LOG" >> "$LOGCAT"

{
  echo "# v2t FD-009 7-day audit — $(date +%Y-%m-%d)"
  echo
  echo "Anchor commit: \`b344c57\` deployed 2026-05-07 ~10:15"
  echo "Log span: $(stat -f %Sm "$LOG")"
  echo "Log file size: $(du -h "$LOG" | cut -f1) (+ $(du -h "$ROT" 2>/dev/null | cut -f1 || echo "n/a") rotation)"
  echo

  echo "## 1. Stability"
  pid=$(pgrep -fx "/Applications/v2t.app/Contents/MacOS/v2t" || true)
  if [ -n "$pid" ]; then
    ps -o pid,etime,rss,%cpu,command -p "$pid" | head -2
  else
    echo "v2t process not running — note for review"
  fi
  echo
  echo "ERROR / panic / fatal count: $(grep -cE 'ERROR|panic|FATAL' "$LOGCAT")"
  echo "WARN count: $(grep -cE 'WARN' "$LOGCAT")"
  echo "Top 5 distinct WARN messages:"
  grep -oE 'WARN\][^"]*' "$LOGCAT" | sort | uniq -c | sort -rn | head -5
  echo

  echo "## 2. M3 lexical biasing"
  echo "Total trigger count: $(grep -c 'lexical biasing context' "$LOGCAT")"
  echo "Context length distribution:"
  grep -oE 'lexical biasing context \([0-9]+ chars\)' "$LOGCAT" \
    | grep -oE '[0-9]+' | sort -n | uniq -c | tail -10
  echo

  echo "## 3. M4 rolling history"
  echo "Total fire count: $(grep -c 'context biasing with' "$LOGCAT")"
  echo "History row distribution (TTL=180s post-0295d13):"
  grep 'context biasing with' "$LOGCAT" \
    | grep -oE 'with [0-9]+ recent' | sort | uniq -c
  echo

  echo "## 4. M2 hybrid trigger"
  echo "Tap (toggle): $(grep -c 'Tap detected' "$LOGCAT")"
  echo "Hold (PTT): $(grep -c 'Hold detected' "$LOGCAT")"
  echo "Toggle hard-timeout fired: $(grep -c 'Toggle hard-timeout' "$LOGCAT")"
  echo

  echo "## 5. CJK collapse / filler regression"
  echo "Scanning history.db transcripts AFTER anchor commit..."
  python3 - <<PY
import sqlite3, re
con = sqlite3.connect("$DB")
rows = con.execute(
    "SELECT id, datetime(timestamp,'unixepoch','localtime'), transcription_text "
    "FROM transcription_history WHERE timestamp > $ANCHOR_TS"
).fetchall()
print(f"Total post-anchor transcripts: {len(rows)}")
sus_rep = re.compile(r"(.)\1{2,}|(.{2})\2{2,}|(.{3})\3{2,}")
mid_filler = re.compile(r"[一-鿿](嗯|呃|啊)[一-鿿]")
rep_hits = filler_hits = 0
for _id, ts, text in rows:
    if not text: continue
    if any(g and ord(g[0]) >= 0x4e00 for m in sus_rep.findall(text) for g in m if g):
        rep_hits += 1
    if mid_filler.search(text):
        filler_hits += 1
print(f"  CJK repetition pattern remnants: {rep_hits}")
print(f"  Mid-sentence CJK filler remnants: {filler_hits}")
PY
  echo

  echo "## 6. A/B sample re-run (5 most recent wavs)"
  if [ ! -x "$EXAMPLE" ]; then
    echo "test_mlx binary missing — rebuild with: cargo build --release --example test_mlx"
  else
    for wav in $(ls -t "$RECDIR"/v2t-*.wav | head -5); do
      fname=$(basename "$wav")
      baseline=$("$EXAMPLE" "$wav" qwen3-asr-06b-8bit 2>&1 | grep "TEXT:" | sed 's/^[[:space:]]*TEXT: //')
      current=$(sqlite3 "$DB" "SELECT transcription_text FROM transcription_history WHERE file_name='$fname' LIMIT 1")
      echo "### $fname"
      echo "- A baseline (no biasing): $baseline"
      echo "- B current (deploy):     $current"
      echo
    done
  fi

  echo "## 7. Action items for next milestone"
  echo "- If M3 trigger count >> 100 with 0 warnings → biasing path is stable"
  echo "- If M4 history hit count grew (vs initial 0/1 at deploy) → TTL 180 was right call"
  echo "- If M2 tap-toggle still 0 → UX hint really is needed"
  echo "- If CJK regression hits > 0 → investigate that pattern"
} | tee "$OUT"

echo
echo "═══════════════════════════════════════════"
echo "  Full report → $OUT"
echo "═══════════════════════════════════════════"
