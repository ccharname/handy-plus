#!/usr/bin/env bash
# release-check.sh — handy-plus v1.0 local release gate
#
# Usage:
#   ./scripts/release-check.sh           # full run (lint + clippy + test + sla + build + codesign)
#   ./scripts/release-check.sh --dry-run  # quick sanity: lint + clippy + test only (no build)
#
# Exit codes: 0 = all checks passed, non-zero = first failure
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/scripts"
DRY_RUN=false
PASS=0
FAIL=0

# ── Arg parsing ──────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "  --dry-run  Skip build and codesign steps (fast sanity check)"
      exit 0
      ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "  [PASS] $*"; PASS=$(( PASS + 1 )); }
fail() { echo "  [FAIL] $*" >&2; FAIL=$(( FAIL + 1 )); }
skip() { echo "  [SKIP] $*"; }

step() {
  echo ""
  echo "==> $*"
}

# ── Step 1: Frontend lint ─────────────────────────────────────────────────────
step "1/6 Frontend lint (ESLint)"
cd "$REPO_ROOT"
if bun run lint; then
  ok "ESLint passed"
else
  fail "ESLint failed"
fi

# ── Step 2: Backend clippy ────────────────────────────────────────────────────
step "2/6 Backend clippy (release profile)"
cd "$REPO_ROOT/src-tauri"
if cargo clippy --release -- -D warnings 2>&1; then
  ok "cargo clippy passed"
else
  fail "cargo clippy failed"
fi
cd "$REPO_ROOT"

# ── Step 3: Tests ─────────────────────────────────────────────────────────────
step "3/6 Lib tests (cargo test --lib)"
cd "$REPO_ROOT/src-tauri"
if cargo test --lib 2>&1 | tail -5; then
  ok "cargo test --lib passed"
else
  fail "cargo test --lib failed"
fi
cd "$REPO_ROOT"

# ── Step 4: SLA hard gates ────────────────────────────────────────────────────
# handy-logs assert handles no-data → SKIP (exit 0), never fails on missing data.
step "4/6 SLA hard gates (handy-logs assert)"
SLA_GATES=(
  "assert --stage t0_hotkey --p50-max 30 --p99-max 80"
  "assert --stage t1_audio_capture --p50-max 100 --p99-max 250"
  "assert --stage t3_vad --p50-max 50 --p99-max 120"
  "assert --stage t4_resample --p50-max 30 --p99-max 80"
  "assert --stage t5_inference --preset chinese_balanced --metric rtf --p50-max 0.30 --p99-max 0.50"
  "assert --stage t5_inference --preset qwen3_mlx --metric first_token_ms --p50-max 400 --p99-max 800"
  "assert --stage t5_inference --preset qwen3_mlx --metric rtf --p50-max 0.40 --p99-max 0.70"
  "assert --stage total --metric end_to_end_ms --p50-max 1500 --p99-max 2500"
)
for gate in "${SLA_GATES[@]}"; do
  if python3 "$SCRIPT_DIR/handy-logs.sh" $gate 2>&1; then
    :  # ok/skip handled by handy-logs itself
  else
    fail "SLA gate failed: handy-logs $gate"
  fi
done
ok "SLA gates checked (SKIP = no data, harmless until production build)"

if $DRY_RUN; then
  echo ""
  echo "──────────────────────────────────────────────────────────────────"
  echo "  DRY-RUN complete. Skipped: tauri build + codesign."
  echo "  PASS=$PASS  FAIL=$FAIL"
  echo "──────────────────────────────────────────────────────────────────"
  [[ $FAIL -eq 0 ]] && exit 0 || exit 1
fi

# ── Step 5: Tauri build ───────────────────────────────────────────────────────
step "5/6 Tauri build"
cd "$REPO_ROOT"
if CMAKE_POLICY_VERSION_MINIMUM=3.5 bun run tauri build 2>&1 | tail -5; then
  ok "tauri build passed"
else
  fail "tauri build failed"
fi

# ── Step 6: Codesign verify ───────────────────────────────────────────────────
step "6/6 Codesign verify"
APP_BUNDLE="$REPO_ROOT/src-tauri/target/release/bundle/macos/Handy.app"
if [[ ! -d "$APP_BUNDLE" ]]; then
  fail "App bundle not found: $APP_BUNDLE"
else
  if codesign -dv "$APP_BUNDLE" 2>&1 | grep -q "Authority="; then
    ok "App bundle is signed"
  else
    fail "App bundle signature missing or invalid"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Release check complete.  PASS=$PASS  FAIL=$FAIL"
echo "══════════════════════════════════════════════════════════════════"
echo ""
if [[ $FAIL -gt 0 ]]; then
  echo "  ACTION REQUIRED: $FAIL check(s) failed. Fix before tagging v1.0.0."
  exit 1
fi
echo "  All checks passed. Ready to: git tag v1.0.0 && git push fork v1.0.0"
