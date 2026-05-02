#!/bin/bash
# Re-sign the freshly built Handy.app with the stable local identity so
# macOS TCC retains its granted permissions (Microphone, Accessibility,
# Speech Recognition) across version upgrades.
#
# Usage:
#   bash scripts/sign-build.sh [path/to/Handy.app]
#
# Default app path: src-tauri/target/release/bundle/macos/Handy.app
#
# Prerequisites:
#   Run scripts/setup-codesign-identity.sh once to create the signing identity.

set -euo pipefail

APP_PATH="${1:-src-tauri/target/release/bundle/macos/Handy.app}"
IDENTITY="Handy+ Local Dev"
ENTITLEMENTS="src-tauri/Entitlements.plist"

# ── Resolve absolute path ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$APP_PATH" != /* ]]; then
    APP_PATH="$REPO_ROOT/$APP_PATH"
fi

if [[ "$ENTITLEMENTS" != /* ]]; then
    ENTITLEMENTS="$REPO_ROOT/$ENTITLEMENTS"
fi

# ── Preflight checks ───────────────────────────────────────────────────────
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: App bundle not found at: $APP_PATH" >&2
    echo "       Build first with: bun run tauri build" >&2
    exit 1
fi

if ! security find-identity -p codesigning -v 2>/dev/null | grep -q "$IDENTITY"; then
    echo "ERROR: Signing identity '$IDENTITY' not found in keychain." >&2
    echo "       Run scripts/setup-codesign-identity.sh first." >&2
    exit 1
fi

if [ ! -f "$ENTITLEMENTS" ]; then
    echo "WARNING: Entitlements.plist not found at $ENTITLEMENTS" >&2
    echo "         Signing without explicit entitlements." >&2
    ENTITLEMENTS_FLAG=""
else
    ENTITLEMENTS_FLAG="--entitlements $ENTITLEMENTS"
fi

# ── Sign ───────────────────────────────────────────────────────────────────
echo "Signing: $APP_PATH"
echo "With:    $IDENTITY"
echo ""

# Sign deep (frameworks, dylibs, helpers first, then the bundle root).
# --options runtime enables Hardened Runtime — required for TCC authority tracking.
# shellcheck disable=SC2086
codesign \
    --force \
    --deep \
    --sign "$IDENTITY" \
    --options runtime \
    $ENTITLEMENTS_FLAG \
    --timestamp=none \
    "$APP_PATH"

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "Verifying signature..."
codesign -dv --verbose=2 "$APP_PATH" 2>&1 | grep -E "^(Authority|TeamIdentifier|Identifier|Format|CodeDirectory|Signature)" || true
echo ""
codesign --verify --deep --strict "$APP_PATH" && echo "Signature OK." || {
    echo "WARNING: Signature verification reported issues (may be non-fatal for local dev)." >&2
}

echo ""
echo "Done. TCC should retain granted permissions across future builds"
echo "as long as the '$IDENTITY' certificate stays in the keychain."
echo ""
echo "If permissions are still re-prompted after the FIRST install of this"
echo "signed build, that is expected (one-time anchor). Subsequent upgrades"
echo "signed with the same identity should not re-prompt."
