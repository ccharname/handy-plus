#!/bin/bash
# Usage: scripts/measure_resource.sh <pid> <duration_sec> <output_csv>
# Samples RSS (KB) + CPU% every 1s for duration_sec seconds.

PID=$1
DUR=${2:-60}
OUT=${3:-/tmp/handy_resource_$PID.csv}

if ! ps -p "$PID" > /dev/null; then
  echo "PID $PID not running" >&2
  exit 1
fi

echo "ts_unix,rss_kb,cpu_pct" > "$OUT"
for i in $(seq 1 "$DUR"); do
  STATS=$(ps -o rss=,pcpu= -p "$PID" 2>/dev/null)
  if [ -z "$STATS" ]; then
    echo "PID $PID exited" >&2
    break
  fi
  TS=$(date +%s)
  echo "$TS,$STATS" | tr -s ' ' ',' >> "$OUT"
  sleep 1
done

echo "Saved: $OUT ($(wc -l < $OUT) lines)"
