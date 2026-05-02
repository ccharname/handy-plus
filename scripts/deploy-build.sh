#!/bin/bash
# Full deploy loop for Handy.app — kill → backup → install → launch → verify.
# Assumes the build at src-tauri/target/release/bundle/macos/Handy.app already
# exists and has been signed by sign-build.sh (run by `bun run tauri:build`).
#
# Use: bash scripts/deploy-build.sh
# Or:  bun run tauri:deploy   (chains build + sign + this)

set -euo pipefail

SRC_APP="src-tauri/target/release/bundle/macos/Handy.app"
DST_APP="/Applications/Handy.app"
TRASH="$HOME/.Trash"

if [ ! -d "$SRC_APP" ]; then
    echo "ERROR: built app not found at $SRC_APP" >&2
    echo "Run 'bun run tauri:build' first." >&2
    exit 1
fi

# Read the new version so the backup name reflects the version being replaced
NEW_VER=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" \
    "$SRC_APP/Contents/Info.plist" 2>/dev/null || echo "unknown")

OLD_VER="unknown"
if [ -d "$DST_APP" ]; then
    OLD_VER=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" \
        "$DST_APP/Contents/Info.plist" 2>/dev/null || echo "unknown")
fi

echo "=== Deploying Handy.app v$NEW_VER (replacing v$OLD_VER) ==="

# Step 1: stop running instance (graceful, then forceful if needed)
if pgrep -f "/Applications/Handy.app/Contents/MacOS/handy" >/dev/null 2>&1; then
    echo "[1/5] Stopping running Handy instance..."
    pkill -f "/Applications/Handy.app/Contents/MacOS/handy" 2>/dev/null || true
    # Wait up to 5s for graceful exit
    for i in {1..10}; do
        if ! pgrep -f "/Applications/Handy.app/Contents/MacOS/handy" >/dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
    # Force-kill if still alive
    if pgrep -f "/Applications/Handy.app/Contents/MacOS/handy" >/dev/null 2>&1; then
        echo "  graceful stop timed out; sending SIGKILL"
        pkill -9 -f "/Applications/Handy.app/Contents/MacOS/handy" 2>/dev/null || true
        sleep 1
    fi
else
    echo "[1/5] No running instance to stop"
fi

# Step 2: backup current install (mv works under /Applications even when rm -rf is blocked)
TS=$(date +%Y%m%d-%H%M%S)
if [ -d "$DST_APP" ]; then
    BACKUP="$TRASH/Handy.app.${OLD_VER}-${TS}"
    echo "[2/5] Backing up current /Applications/Handy.app -> $BACKUP"
    mv "$DST_APP" "$BACKUP"
else
    echo "[2/5] No existing install to back up"
fi

# Step 3: install new build
echo "[3/5] Installing v$NEW_VER -> $DST_APP"
cp -R "$SRC_APP" "$DST_APP"

# Step 4: launch (hidden so it goes to tray immediately)
echo "[4/5] Launching..."
open "$DST_APP" --args --start-hidden

# Step 5: verify (give the launcher 3s to spawn the helper)
sleep 3
RUNNING_VER=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" \
    "$DST_APP/Contents/Info.plist" 2>/dev/null || echo "unknown")

if pgrep -f "/Applications/Handy.app/Contents/MacOS/handy" >/dev/null 2>&1; then
    PID=$(pgrep -f "/Applications/Handy.app/Contents/MacOS/handy" | head -1)
    echo "[5/5] OK — Handy v$RUNNING_VER running (pid $PID)"
else
    echo "[5/5] WARN — Handy.app installed but process not detected; check Console" >&2
    exit 1
fi

# Verify code signing (helps debug 'why is TCC re-prompting?' regressions)
AUTH=$(codesign -dv "$DST_APP" 2>&1 | grep '^Authority=' | head -1 || echo "")
if [ -n "$AUTH" ]; then
    echo "      $AUTH"
fi

echo ""
echo "Deploy complete. Old v$OLD_VER kept in $TRASH for rollback."
