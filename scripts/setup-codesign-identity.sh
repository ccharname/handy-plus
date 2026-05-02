#!/bin/bash
# One-time setup: create a self-signed Code Signing identity in the user's
# login keychain so all subsequent Handy+ builds can be signed with the same
# authority.
#
# macOS TCC tracks the signing *authority* (cert SHA1) rather than the binary
# hash when hardenedRuntime is used. Re-signed builds of the same app keep
# their already-granted permissions (Microphone, Accessibility, Speech
# Recognition) so users are not re-prompted on every upgrade.
#
# Run once per developer machine:
#   bash scripts/setup-codesign-identity.sh
#
# Verify afterward:
#   security find-identity -p codesigning -v

set -euo pipefail

IDENTITY_NAME="Handy+ Local Dev"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

# ── Check if already present and valid ────────────────────────────────────
if security find-identity -p codesigning -v 2>/dev/null | grep -q "$IDENTITY_NAME"; then
    echo "Identity '$IDENTITY_NAME' already exists and is valid — skipping creation."
    echo ""
    security find-identity -p codesigning -v | grep "$IDENTITY_NAME" || true
    exit 0
fi

echo "Creating self-signed Code Signing identity '$IDENTITY_NAME'..."
echo ""
echo "IMPORTANT: macOS will show a system dialog asking for your login"
echo "keychain password to trust this certificate. This is required once."
echo ""

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# ── Generate private key + self-signed certificate ─────────────────────────
cat > "$WORKDIR/cert.conf" <<OPENSSL_EOF
[ req ]
distinguished_name = req_distinguished_name
prompt             = no
x509_extensions    = v3_ext

[ req_distinguished_name ]
CN = $IDENTITY_NAME
O  = Handy+ Local Dev
C  = US

[ v3_ext ]
keyUsage         = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
basicConstraints = critical, CA:false
subjectKeyIdentifier = hash
OPENSSL_EOF

openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$WORKDIR/key.pem" \
    -out    "$WORKDIR/cert.pem" \
    -days   3650 \
    -config "$WORKDIR/cert.conf" 2>/dev/null

# ── Step 1: Import certificate ─────────────────────────────────────────────
echo "Step 1/4: Importing certificate..."
security import "$WORKDIR/cert.pem" -k "$KEYCHAIN" 2>&1

# ── Step 2: Import private key ─────────────────────────────────────────────
# -T flags pre-authorise codesign and security tools so macOS does not
# prompt for keychain access every time we sign a build.
echo "Step 2/4: Importing private key..."
security import "$WORKDIR/key.pem" \
    -k "$KEYCHAIN" \
    -T /usr/bin/codesign \
    -T /usr/bin/security 2>&1

# ── Step 3: Trust the certificate for code signing ─────────────────────────
# This will trigger a macOS GUI dialog asking for the keychain password.
# It is required to make the identity show as "valid" for codesigning.
echo "Step 3/4: Trusting certificate for code signing..."
echo "(A system password dialog will appear — enter your login password)"
security add-trusted-cert \
    -d \
    -r trustRoot \
    -p codeSign \
    -k "$KEYCHAIN" \
    "$WORKDIR/cert.pem" 2>&1 || {
    echo ""
    echo "WARNING: Could not set trust automatically."
    echo ""
    echo "MANUAL FALLBACK — run this command and enter your password when prompted:"
    echo "  security add-trusted-cert -d -r trustRoot -p codeSign -k \"$KEYCHAIN\" <path-to-cert>"
    echo ""
    echo "Or use Keychain Access:"
    echo "  1. Open Keychain Access"
    echo "  2. Find '$IDENTITY_NAME' under 'My Certificates' or 'Certificates'"
    echo "  3. Double-click → Trust → Code Signing → 'Always Trust'"
    echo "  4. Enter your password when prompted"
    echo ""
    echo "Then re-run this script (Step 3/4 will be skipped automatically)."
    exit 1
}

# ── Step 4: Set partition list ─────────────────────────────────────────────
# Allows codesign to use the private key without a passphrase prompt.
echo "Step 4/4: Configuring keychain access for codesign..."
security set-key-partition-list \
    -S "apple-tool:,apple:,codesign:" \
    -s \
    "$KEYCHAIN" >/dev/null 2>&1 || {
    echo ""
    echo "WARNING: Could not set partition list automatically."
    echo "codesign may prompt for keychain access during builds."
    echo "If so, run:"
    echo "  security set-key-partition-list -S 'apple-tool:,apple:,codesign:' -s -k '<your-password>' '$KEYCHAIN'"
}

echo ""
echo "Identity '$IDENTITY_NAME' created and trusted successfully."
echo ""
echo "Verify with:"
echo "  security find-identity -p codesigning -v"
echo ""
echo "Next steps:"
echo "  1. Build Handy:  bun run tauri:build"
echo "     (runs: bun run tauri build && bash scripts/sign-build.sh)"
echo "  2. Install and grant permissions ONCE in System Settings → Privacy & Security"
echo "  3. All future builds will reuse this identity — no more re-prompts."
