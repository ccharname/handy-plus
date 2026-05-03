# mlx-audio Sidecar Recon (Round B prework)

This directory contains the recon artifacts produced overnight on
2026-05-04 ahead of a possible "Round B" — adopting **mlx-audio** as a
unified Python sidecar backend for true streaming ASR (the missing piece
that the existing sherpa-onnx Rust path cannot deliver).

Round B was **not implemented** in the night-shift window because the
clean engineering total (~7-9 days) physically does not fit a 7-hour
window. Hatching it half-done would have left a polluted commit history
and a non-shippable sidecar. Phase B1 will start as a dedicated workstream
once zheng signs off on the plan in REPORT.md.

## Files

- **`REPORT.md`** — Full feasibility report from the recon agent. Sections
  cover mlx-audio current state, sidecar embedding options (PyOxidizer
  dead, uv standalone Python recommended), live PoC results (Voxtral
  Realtime confirmed truly streaming, Qwen3-ASR pseudo-streaming via
  mlx-audio), implementation plan in three phases, and a conditional
  GO recommendation.

- **`poc/poc_voxtral_stream.py`** — Standalone PoC that proves Voxtral
  Realtime via `VoxtralStreamingSession.feed()` + `step()` actually
  streams tokens proportional to audio time. Re-runnable.

- **`poc/poc_qwen3_stream.py`** — Standalone PoC that demonstrates
  Qwen3-ASR via mlx-audio is pseudo-streaming (batch compute then
  token-by-token yield). Re-runnable.

## Re-running the PoC

The recon agent left a working venv at `/tmp/handy-mlx-recon/poc-venv/`
with mlx-audio 0.4.3 installed. The Voxtral 4-bit model weights
(2.9 GB) are cached at
`~/.cache/huggingface/hub/models--mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit/`.

```bash
source /tmp/handy-mlx-recon/poc-venv/bin/activate
cd ~/Developer/handy/docs/dev/mlx-recon/poc
python poc_voxtral_stream.py /tmp/qwen3_bench_zh.wav
```

If the temp venv has been cleaned, recreate it:

```bash
mkdir -p /tmp/handy-mlx-recon && cd /tmp/handy-mlx-recon
uv venv poc-venv --python 3.12
source poc-venv/bin/activate
uv pip install "mlx-audio[stt]"
```

## Critical preconditions before Phase B1 starts

These two items MUST be resolved before any sidecar code is written. They
are documented inline in REPORT.md Section 5 but called out here because
each is a genuine show-stopper if missed.

### 1. Entitlement: `com.apple.security.cs.allow-unsigned-executable-memory`

**Status (verified 2026-05-04 night):** MISSING from
`src-tauri/entitlements.mac.plist`. The current plist only declares
`microphone` and `audio-input`.

MLX JIT-compiles Metal shaders at runtime. Without this entitlement, the
sidecar process will hard-crash on first model use under hardened
runtime.

**Action:** Before Phase B1, add the key to `entitlements.mac.plist` and
run a full `bun run tauri:deploy` to confirm the existing Whisper Metal
+ sherpa-onnx paths still work under the loosened entitlement. This is
a 1-hour pre-check.

### 2. Bundle strategy — bundled vs unbundled

**Bundle delta:** sidecar Python interpreter + mlx-audio +
deps = ~447 MB. With the existing `.app` (40 MB), the bundled total is
~490 MB. Voxtral weights (~2.9 GB) and Qwen3-ASR weights (~966 MB) are
download-on-demand, not bundled.

**Decision needed:** ship the sidecar inside the .app for zero-config
power-user UX, or expect users to `pip install mlx-audio` themselves
to keep the bundle small. zheng must decide before B1 scope is locked.
For handy-plus's audience (power users dictating in Chinese), the
recon agent recommends bundled.

## Round B status as of this directory's creation

- Recon: ✅ done
- PoC: ✅ Voxtral Realtime confirmed truly streaming
- Plan: ✅ in REPORT.md (Phase B1: ~4-5d, B2: ~3-4d, B3: ~2-3d each)
- Preconditions: ⚠️ entitlement missing + bundle strategy undecided
- Implementation: ⏸ blocked on zheng's GO

## Strategic note on mlx-audio-swift

REPORT.md mentions `Blaizzy/mlx-audio-swift` as a possible Phase C —
native Swift SDK with the same Voxtral / Qwen3-ASR streaming, no Python.
Would integrate via a thin C FFI bridge. Estimated 1-2 weeks longer than
the Python sidecar but eliminates the bundle/signing complexity. Worth
considering if the Python sidecar's 490 MB bundle proves user-hostile.
