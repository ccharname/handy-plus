# Code Signing Setup for Handy+ (macOS)

## Problem

macOS TCC (Transparency, Consent, and Control) tracks whether an app has been
granted permissions such as Microphone, Accessibility, and Speech Recognition.
When an app is signed with an **ad-hoc identity** (`-`), macOS records the grant
against that specific binary hash (cdhash). Every new build produces a different
binary → different cdhash → TCC treats it as a new, untrusted app → permissions
dialog re-appears on every upgrade.

## Solution

Sign every build with a **stable self-signed identity** stored in your login
keychain. macOS TCC tracks the signing *authority* (the certificate's SHA1
fingerprint) rather than the binary hash. As long as each build is signed by the
same certificate, TCC preserves the granted permissions across version upgrades.

## One-Time Setup (per developer machine)

Run **once** to create the signing identity:

```bash
bash scripts/setup-codesign-identity.sh
```

This script:
1. Generates a `rsa:2048` self-signed certificate with `codeSigning` EKU, valid
   for 10 years.
2. Imports it into `~/Library/Keychains/login.keychain-db`.
3. Pre-authorises `/usr/bin/codesign` so builds do not prompt for keychain
   access.

Verify the identity was created:

```bash
security find-identity -p codesigning -v
# Should include: "Handy+ Local Dev"
```

> **If the script fails on `set-key-partition-list`**, your keychain may be
> locked. Unlock it first:
> ```bash
> security unlock-keychain ~/Library/Keychains/login.keychain-db
> bash scripts/setup-codesign-identity.sh
> ```

> **Fallback (GUI):** Open *Keychain Access* → *Certificate Assistant* →
> *Create a Certificate* → Name: `Handy+ Local Dev`, Type: *Code Signing*,
> let it self-sign. Then re-run the script (it will skip creation and only
> set the partition list).

## Building and Signing

Use the combined script instead of plain `tauri build`:

```bash
bun run tauri:build
# equivalent to: bun run tauri build && bash scripts/sign-build.sh
```

Or run steps manually:

```bash
bun run tauri build
bash scripts/sign-build.sh
# optionally pass a custom app path:
bash scripts/sign-build.sh path/to/Handy.app
```

## First Installation After Setup

The **very first** build you install after running setup will still trigger the
macOS permissions dialog (Microphone, Accessibility, etc.) — this is unavoidable
because TCC has no prior record for this signing authority. Grant permissions
once in *System Settings → Privacy & Security*.

**All subsequent upgrades** signed with the same `Handy+ Local Dev` certificate
will not re-prompt.

## tauri.conf.json

`src-tauri/tauri.conf.json` already references the identity:

```json
"macOS": {
  "signingIdentity": "Handy+ Local Dev",
  "hardenedRuntime": true,
  "entitlements": "Entitlements.plist"
}
```

Tauri passes this value to `codesign` during the bundling step. `sign-build.sh`
then re-signs with `--force --deep` to ensure nested frameworks and helpers are
also covered.

## Important Caveats

### macOS 15 (Sequoia) behaviour
Apple tightened TCC in macOS 15. Some permission types may still re-prompt when
the **Team Identifier** is absent (self-signed certs do not carry a Developer ID
team). If you observe this, the nuclear fallback is:

```bash
# As an admin, grant permissions without a prompt (once per machine).
# Replace com.pais.handy with the actual bundle ID if changed.
sudo tccutil reset Microphone com.pais.handy
sudo tccutil reset Accessibility com.pais.handy
sudo tccutil reset SpeechRecognition com.pais.handy
# Then launch Handy once and grant via the dialog — it will stick.
```

### Certificate lifetime
The generated certificate is valid for 10 years. If it expires or is deleted,
re-run `setup-codesign-identity.sh`. Users will see permissions re-prompted
**once** after switching to the new certificate, then never again.

### Sharing with other developers
Each developer runs `setup-codesign-identity.sh` on their own machine. The
resulting certificates are **different** (different key pairs), so they will each
need a one-time permission grant after the first build. This is unavoidable with
self-signed certs.

For a shared team cert (zero re-prompts across machines), you would need an
Apple Developer Program membership and a Developer ID Application certificate.

### Gatekeeper
Self-signed apps are **not notarised**. Gatekeeper will block the app when
opened on a machine other than the one that built it. This setup is intended for
**developer use only** (personal builds, local testing). Distribution builds
should use a proper Developer ID certificate.
