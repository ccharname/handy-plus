# Apple Speech Permission State Test Plan

> Version: v0.8.12 | Platform: macOS 10.15+ (SFSpeechRecognizer)
> Related: E-task (v0.8.7) — 4-class error distinction + GCD timer guard
> Script: `scripts/test_apple_perm_states.sh`

## 1. SFSpeechRecognizerAuthorizationStatus Raw Values

| raw value | enum case      | Handy error class | Description |
|:---------:|----------------|-------------------|-------------|
| 0         | `notDetermined`| AUTH_TIMEOUT or none | App has never requested permission; first transcription triggers the system dialog. |
| 1         | `restricted`   | PERM_DENIED       | Device policy (MDM / Screen Time / Configuration Profile) prohibits speech recognition. User cannot change this. |
| 2         | `denied`       | PERM_DENIED       | User explicitly denied permission (or toggled Handy OFF in System Settings). |
| 3         | `authorized`   | — (no error)      | User authorized speech recognition for Handy. Normal transcription proceeds. |

`-1` / `-2` are returned by `apple_speech_get_auth_status()` for macOS < 10.15 or unknown future enum cases (`SpeechAuthStatus::Unsupported`).

## 2. Error Classification in Code

Two layers work together:

### Layer 1: Swift (`src-tauri/swift/apple_speech.swift`)

`transcribeImpl()` (line 47–245) checks permission before starting recognition:

```swift
// line 96: fast path — no dialog if already determined
var authStatus = SFSpeechRecognizer.authorizationStatus()

// line 97-109: notDetermined → show dialog with 5-second guard
if authStatus == .notDetermined {
    SFSpeechRecognizer.requestAuthorization { status in ... }
    if authSemaphore.wait(timeout: .now() + .seconds(5)) == .timedOut {
        // → AUTH_TIMEOUT: prefix (line 104-108)
    }
}

// line 111-134: switch on resolved status
case .denied:    // → PERM_DENIED: prefix (line 115-118)
case .restricted: // → PERM_DENIED: prefix (line 119-121)
case .notDetermined: // → PERM_DENIED: prefix (line 124-127)
```

GCD timer guard (lines 203-217): independently fires after `timeout_ms` ms, sets `box.error = "TIMEOUT: ..."` and signals the semaphore — ensuring the thread is never permanently blocked.

Engine errors (line 238-241): any unclassified error is prefixed with `"ENGINE: "`.

### Layer 2: Rust (`src-tauri/src/apple_speech.rs`)

`parse_apple_speech_error()` (lines 54-67) strips the prefix and returns a typed enum:

```rust
AppleSpeechError::PermissionDenied  // "PERM_DENIED: ..."
AppleSpeechError::AuthTimeout       // "AUTH_TIMEOUT: ..."
AppleSpeechError::Timeout           // "TIMEOUT: ..."
AppleSpeechError::Engine            // "ENGINE: ..." or unprefixed (legacy)
```

The `Display` impl formats each variant with a user-friendly message including System Settings navigation hints for permission errors.

The Tauri command `get_speech_recognition_permission` (`src-tauri/src/commands/audio.rs` line 156) queries the status via `apple_speech_get_auth_status()` FFI without triggering any dialog or recording — safe to call at any time.

## 3. Per-State Test Matrix

### 3.1 authorized (status=3)

**How to reach:** System Settings → Privacy & Security → Speech Recognition → Handy ON

**Expected behavior:**
- `get_speech_recognition_permission` returns `"authorized"`
- `apple_speech_get_auth_status()` FFI returns `3`
- `transcribeImpl` skips the `notDetermined` branch entirely (line 96 short-circuits)
- Transcription proceeds normally; result text returned with `success=1`
- No error logged; no error class emitted

**Test result: PASS (bench evidence)**

Bench file: `benchmark/results/v0.8.8/apple_native_2026-05-02T09-11-38Z.json`
- 38 items processed, **0 errors**, no hang
- `cold_start_latency_ms=3608`, `steady_p50_latency_ms=748`, `punc_density=0.0811`
- GCD timer never fired (all items completed within timeout)
- Log window 09:11–09:12Z: 41 `punc_zh: applied punctuation` entries; 0 ERROR entries

Live verification: `swift -e 'import Speech; print(SFSpeechRecognizer.authorizationStatus().rawValue)'` → `3` (confirmed 2026-05-01).

### 3.2 denied (status=2)

**How to reach:** System Settings → Privacy & Security → Speech Recognition → toggle Handy OFF

**Expected behavior:**
- `get_speech_recognition_permission` returns `"denied"`
- `transcribeImpl` hits the `.denied` switch branch (swift line 115)
- Returns immediately with `error_message = "PERM_DENIED: Speech recognition authorization denied. Please enable in System Settings → Privacy & Security → Speech Recognition."`
- `parse_apple_speech_error` maps to `AppleSpeechError::PermissionDenied`
- Handy log: `[ERROR] Apple Speech permission not granted: Speech recognition authorization denied...`
- No GCD timer involved (returns before recognition task starts)

**Test result: pending manual**

Manual checklist:
1. Toggle Handy OFF in System Settings → Privacy & Security → Speech Recognition
2. Verify `swift -e 'import Speech; print(SFSpeechRecognizer.authorizationStatus().rawValue)'` → `2`
3. Run `bash scripts/test_apple_perm_states.sh` and follow Case 2 prompts
4. Trigger a recording via Handy hotkey
5. Check `~/Library/Logs/com.pais.handy/handy.log` for `PERM_DENIED` entry
6. Re-enable Handy in System Settings after test

### 3.3 notDetermined (status=0)

**How to reach:** `sudo tccutil reset Speech com.pais.handy` (separate terminal), then restart Handy

**Expected behavior (three sub-scenarios):**

| User action at dialog | Expected error class | Expected log |
|----------------------|---------------------|--------------|
| Clicks "Allow"       | none                | normal transcription result |
| Clicks "Don't Allow" | PERM_DENIED         | `PERM_DENIED: Speech recognition authorization denied.` |
| No dialog shown / no user present (headless) | AUTH_TIMEOUT | `AUTH_TIMEOUT: Speech recognition authorization dialog timed out (5s).` |

The 5-second authorization timeout guard is implemented at `apple_speech.swift` lines 102-108.

**Test result: pending manual**

Manual checklist:
1. In a separate terminal: `sudo tccutil reset Speech com.pais.handy`
2. Quit and restart Handy (the app must re-initialize to see status=0)
3. Verify `swift -e 'import Speech; print(SFSpeechRecognizer.authorizationStatus().rawValue)'` → `0`
4. Run `bash scripts/test_apple_perm_states.sh` and follow Case 3 prompts
5. For "Allow" sub-scenario: click Allow when dialog appears; verify transcription succeeds
6. For "Don't Allow" sub-scenario: repeat `sudo tccutil reset` + restart; click Don't Allow; verify `PERM_DENIED` in log
7. For AUTH_TIMEOUT sub-scenario: not easily reproducible without headless environment; code path is `apple_speech.swift:103-108`

### 3.4 restricted (status=1)

**How to reach:** MDM Configuration Profile or Screen Time with Speech Recognition restriction

**Expected behavior:**
- `get_speech_recognition_permission` returns `"restricted"`
- `transcribeImpl` hits the `.restricted` switch branch (swift line 119)
- Returns with `error_message = "PERM_DENIED: Speech recognition is restricted on this device."`
- Maps to `AppleSpeechError::PermissionDenied` (same as denied)
- User-facing message clearly states "restricted" so users know they cannot fix it themselves

**Test result: DEFERRED**

Not testable on a personal Mac without:
- Enterprise MDM enrollment (Jamf, Mosyle, etc.)
- Screen Time restriction on Speech Recognition
- A Configuration Profile that disallows Speech Recognition

Code path is verified by source inspection:
- `apple_speech.swift` lines 119-121
- `apple_speech.rs` line 55 (PERM_DENIED prefix catch)
- `get_auth_status()` raw value 1 → `SpeechAuthStatus::Restricted` (`apple_speech.rs` line 131)

## 4. Error Class → User Message Mapping

| Class            | User-visible message (via `Display` impl) |
|------------------|-------------------------------------------|
| `PermissionDenied` | "Apple Speech permission not granted: {msg}. Please authorize in System Settings → Privacy & Security → Speech Recognition." |
| `AuthTimeout`    | "Apple Speech authorization timed out: {msg}" |
| `Timeout`        | "Apple Speech timed out: {msg}. The recognizer may be stuck on first-use; please try again." |
| `Engine`         | "Apple Speech engine error: {msg}" |

Source: `apple_speech.rs` lines 32-50.

## 5. get_speech_recognition_permission Tauri Command

The frontend can query permission status at any time without side effects:

```typescript
// bindings.ts (auto-generated)
// Returns: "authorized" | "denied" | "restricted" | "not_determined" | "unsupported"
const status = await commands.getSpeechRecognitionPermission();
```

Rust implementation: `src-tauri/src/commands/audio.rs` lines 152-172.
Registered in: `src-tauri/src/lib.rs` line 466.

This command is safe to call during app initialization to decide whether to show a permission prompt UI or bypass the Apple Speech engine entirely.

## 6. TEST_REPORT §5.2 T-13.x Checklist

| Sub-test | Method | Result |
|----------|--------|--------|
| T-13.1 authorized → transcription passes | bench (auto) | **PASS** — `apple_native_2026-05-02T09-11-38Z.json`: 38 items, 0 errors, steady_p50=748ms |
| T-13.2 denied → PERM_DENIED error | manual | **pending manual** — see §3.2 checklist |
| T-13.3 notDetermined → dialog or AUTH_TIMEOUT | manual | **pending manual** — see §3.3 checklist |
| T-13.4 authorized, GCD timer no hang | bench (auto) | **PASS (inherited v0.8.8)** — all 38 apple_native items completed; timer never fired |
| T-13.5 4-class error distinction implemented | code review | **code-verified** — `parse_apple_speech_error` at `apple_speech.rs:54-67`; Swift prefixes at `apple_speech.swift:104-133,207-210,238-241` |
| T-13.6 get_speech_recognition_permission command | code review | **code-verified** — `commands/audio.rs:156-172`; FFI `apple_speech_get_auth_status` maps raw 0-3 correctly |
| T-13.7 restricted → PERM_DENIED (same class as denied) | deferred | **DEFERRED** — MDM/Config Profile required; code path verified at `apple_speech.swift:119-121` |
