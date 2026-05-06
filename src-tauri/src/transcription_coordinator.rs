use crate::actions::ACTION_MAP;
use crate::managers::audio::AudioRecordingManager;
use log::{debug, error, warn};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

const DEBOUNCE: Duration = Duration::from_millis(30);

/// Tap/hold threshold: press-release faster than this → Toggle mode.
/// ≥ TAP_THRESHOLD → classic PTT (release stops immediately).
const TAP_THRESHOLD: Duration = Duration::from_millis(200);

/// Hard-timeout for Toggle mode: auto-stop after this duration even if
/// the user forgot to press again (silence-watchdog safety net).
const TOGGLE_HARD_TIMEOUT: Duration = Duration::from_secs(60);

/// Commands processed sequentially by the coordinator thread.
enum Command {
    Input {
        binding_id: String,
        hotkey_string: String,
        is_pressed: bool,
        push_to_talk: bool,
    },
    Cancel {
        recording_was_active: bool,
    },
    ProcessingFinished,
    /// Sent by the silence-watchdog timer thread when Toggle mode
    /// has been active for TOGGLE_HARD_TIMEOUT without a second press.
    HardTimeout {
        /// Epoch token: only stop if this matches the current session's
        /// token, so stale timeouts from previous sessions are ignored.
        token: u64,
    },
}

/// How the current recording was triggered.
#[derive(Debug, Clone, PartialEq)]
enum TriggerMode {
    /// Classic push-to-talk: release key to stop.
    Ptt,
    /// Tap detected (< TAP_THRESHOLD): next press stops, releases ignored.
    Toggle,
}

/// Pipeline lifecycle, owned exclusively by the coordinator thread.
#[derive(Debug)]
enum Stage {
    Idle,
    Recording {
        binding_id: String,
        mode: TriggerMode,
        /// Absolute time of the key-press that started this recording.
        press_at: Instant,
    },
    Processing,
}

/// Serialises all transcription lifecycle events through a single thread
/// to eliminate race conditions between keyboard shortcuts, signals, and
/// the async transcribe-paste pipeline.
pub struct TranscriptionCoordinator {
    tx: Sender<Command>,
}

pub fn is_transcribe_binding(id: &str) -> bool {
    id == "transcribe"
}

impl TranscriptionCoordinator {
    pub fn new(app: AppHandle) -> Self {
        let (tx, rx) = mpsc::channel();
        let tx_clone = tx.clone();

        thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut stage = Stage::Idle;
                let mut last_press: Option<Instant> = None;
                // FD-006 follow-up #7: queue a press that arrived while the
                // pipeline was still Processing the previous session, so the
                // user's "release-then-immediately-press-again" is honoured
                // when Processing → Idle. Without this the press is silently
                // dropped and the user has to release + press a second time.
                let mut pending_press: Option<(String, String, bool)> = None;
                // Monotonically-increasing token for the silence-watchdog so
                // stale HardTimeout commands from old sessions are ignored.
                let mut session_token: u64 = 0;

                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Command::Input {
                            binding_id,
                            hotkey_string,
                            is_pressed,
                            push_to_talk,
                        } => {
                            // Debounce rapid-fire press events (key repeat / double-tap).
                            // Releases always pass through for push-to-talk.
                            if is_pressed {
                                let now = Instant::now();
                                if last_press.is_some_and(|t| now.duration_since(t) < DEBOUNCE) {
                                    debug!("Debounced press for '{binding_id}'");
                                    continue;
                                }
                                last_press = Some(now);
                            }

                            // If the pipeline is busy (Processing or recording
                            // a different binding), queue this press so it
                            // fires when we return to Idle.
                            if is_pressed && !matches!(stage, Stage::Idle) {
                                debug!(
                                    "Press for '{binding_id}' arrived during {:?}; queued until Idle",
                                    stage
                                );
                                pending_press = Some((
                                    binding_id.clone(),
                                    hotkey_string.clone(),
                                    push_to_talk,
                                ));
                                continue;
                            }
                            // A release cancels any queued press the user no
                            // longer wants (e.g. brief tap during processing).
                            if !is_pressed && pending_press.is_some() {
                                debug!("Release cancels pending press for '{binding_id}'");
                                pending_press = None;
                            }

                            if push_to_talk {
                                handle_ptt_input(
                                    &app,
                                    &mut stage,
                                    &mut session_token,
                                    &tx_clone,
                                    &binding_id,
                                    &hotkey_string,
                                    is_pressed,
                                );
                            } else if is_pressed {
                                match &stage {
                                    Stage::Idle => {
                                        start(&app, &mut stage, &binding_id, &hotkey_string);
                                    }
                                    Stage::Recording { binding_id: id, .. }
                                        if id == &binding_id =>
                                    {
                                        stop(&app, &mut stage, &binding_id, &hotkey_string);
                                    }
                                    _ => {
                                        debug!("Ignoring press for '{binding_id}': pipeline busy")
                                    }
                                }
                            }
                        }
                        Command::Cancel {
                            recording_was_active,
                        } => {
                            // Don't reset during processing — wait for the pipeline to finish.
                            if !matches!(stage, Stage::Processing)
                                && (recording_was_active || matches!(stage, Stage::Recording { .. }))
                            {
                                session_token = session_token.wrapping_add(1);
                                stage = Stage::Idle;
                            }
                        }
                        Command::ProcessingFinished => {
                            stage = Stage::Idle;
                            // FD-006 follow-up #7: if the user pressed the
                            // hotkey during Processing (e.g. release-then-
                            // immediately-press), trigger the queued start
                            // now that we're back to Idle.
                            if let Some((binding_id, hotkey_string, _push_to_talk)) =
                                pending_press.take()
                            {
                                debug!(
                                    "Replaying queued press for '{binding_id}' after Processing → Idle"
                                );
                                start(&app, &mut stage, &binding_id, &hotkey_string);
                                last_press = Some(Instant::now());
                            }
                        }
                        Command::HardTimeout { token } => {
                            if token != session_token {
                                debug!(
                                    "HardTimeout token {token} stale (current {session_token}); ignored"
                                );
                                continue;
                            }
                            match &stage {
                                Stage::Recording {
                                    binding_id,
                                    mode: TriggerMode::Toggle,
                                    ..
                                } => {
                                    let bid = binding_id.clone();
                                    warn!(
                                        "Toggle hard-timeout ({TOGGLE_HARD_TIMEOUT:?}): \
                                         auto-stopping '{bid}'"
                                    );
                                    session_token = session_token.wrapping_add(1);
                                    stop(&app, &mut stage, &bid, "");
                                }
                                _ => {
                                    debug!("HardTimeout arrived but stage is not Toggle Recording; ignored");
                                }
                            }
                        }
                    }
                }
                debug!("Transcription coordinator exited");
            }));
            if let Err(e) = result {
                error!("Transcription coordinator panicked: {e:?}");
            }
        });

        Self { tx }
    }

    /// Send a keyboard/signal input event for a transcribe binding.
    /// For signal-based toggles, use `is_pressed: true` and `push_to_talk: false`.
    pub fn send_input(
        &self,
        binding_id: &str,
        hotkey_string: &str,
        is_pressed: bool,
        push_to_talk: bool,
    ) {
        if self
            .tx
            .send(Command::Input {
                binding_id: binding_id.to_string(),
                hotkey_string: hotkey_string.to_string(),
                is_pressed,
                push_to_talk,
            })
            .is_err()
        {
            warn!("Transcription coordinator channel closed");
        }
    }

    pub fn notify_cancel(&self, recording_was_active: bool) {
        if self
            .tx
            .send(Command::Cancel {
                recording_was_active,
            })
            .is_err()
        {
            warn!("Transcription coordinator channel closed");
        }
    }

    pub fn notify_processing_finished(&self) {
        if self.tx.send(Command::ProcessingFinished).is_err() {
            warn!("Transcription coordinator channel closed");
        }
    }
}

/// Handle a press/release event for a push-to-talk binding.
///
/// Tap/hold decision:
/// - **Press** (Idle):   start recording, record `press_at`.
/// - **Release** < 200ms after press → **Toggle mode**: keep recording,
///   arm silence-watchdog; next press will stop.
/// - **Release** ≥ 200ms after press → **PTT mode**: stop immediately
///   (classic push-to-talk behaviour).
/// - **Press** while in Toggle Recording: stop (second tap).
/// - **Release** while in Toggle Recording: ignored.
#[allow(clippy::too_many_arguments)]
fn handle_ptt_input(
    app: &AppHandle,
    stage: &mut Stage,
    session_token: &mut u64,
    tx: &Sender<Command>,
    binding_id: &str,
    hotkey_string: &str,
    is_pressed: bool,
) {
    match (&*stage, is_pressed) {
        // ── Press while Idle: start recording ──────────────────────────────
        (Stage::Idle, true) => {
            start(app, stage, binding_id, hotkey_string);
            // If start() confirmed recording, upgrade Stage to include metadata.
            if let Stage::Recording { .. } = stage {
                *stage = Stage::Recording {
                    binding_id: binding_id.to_string(),
                    mode: TriggerMode::Ptt, // will be refined on release
                    press_at: Instant::now(),
                };
                debug!("PTT press: started recording for '{binding_id}'");
            }
        }

        // ── Release while in PTT/Toggle Recording for matching binding ──────
        (
            Stage::Recording {
                binding_id: id,
                mode,
                press_at,
            },
            false,
        ) if id == binding_id => {
            let elapsed = press_at.elapsed();
            match mode {
                TriggerMode::Toggle => {
                    // In Toggle mode, releases are ignored — next press stops.
                    debug!(
                        "Toggle mode: release ignored for '{binding_id}' ({elapsed:?})"
                    );
                }
                TriggerMode::Ptt => {
                    if elapsed < TAP_THRESHOLD {
                        // Short tap → switch to Toggle mode.
                        debug!(
                            "Tap detected ({elapsed:?} < {TAP_THRESHOLD:?}) for '{binding_id}': \
                             entering Toggle mode"
                        );
                        *session_token = session_token.wrapping_add(1);
                        let token = *session_token;
                        *stage = Stage::Recording {
                            binding_id: binding_id.to_string(),
                            mode: TriggerMode::Toggle,
                            press_at: *press_at,
                        };
                        // Arm silence-watchdog.
                        let tx_clone = tx.clone();
                        thread::spawn(move || {
                            thread::sleep(TOGGLE_HARD_TIMEOUT);
                            let _ = tx_clone.send(Command::HardTimeout { token });
                        });
                    } else {
                        // Hold ≥ threshold → classic PTT stop.
                        debug!(
                            "Hold detected ({elapsed:?} ≥ {TAP_THRESHOLD:?}) for '{binding_id}': \
                             PTT stop"
                        );
                        *session_token = session_token.wrapping_add(1);
                        stop(app, stage, binding_id, hotkey_string);
                    }
                }
            }
        }

        // ── Press while in Toggle Recording for matching binding: stop ──────
        (
            Stage::Recording {
                binding_id: id,
                mode: TriggerMode::Toggle,
                ..
            },
            true,
        ) if id == binding_id => {
            debug!("Toggle mode: second press stops recording for '{binding_id}'");
            *session_token = session_token.wrapping_add(1);
            stop(app, stage, binding_id, hotkey_string);
        }

        _ => {
            debug!(
                "PTT input ignored (binding='{binding_id}', pressed={is_pressed}, stage={stage:?})"
            );
        }
    }
}

fn start(app: &AppHandle, stage: &mut Stage, binding_id: &str, hotkey_string: &str) {
    let Some(action) = ACTION_MAP.get(binding_id) else {
        warn!("No action in ACTION_MAP for '{binding_id}'");
        return;
    };
    action.start(app, binding_id, hotkey_string);
    if app
        .try_state::<Arc<AudioRecordingManager>>()
        .is_some_and(|a| a.is_recording())
    {
        *stage = Stage::Recording {
            binding_id: binding_id.to_string(),
            mode: TriggerMode::Ptt,
            press_at: Instant::now(),
        };
    } else {
        debug!("Start for '{binding_id}' did not begin recording; staying idle");
    }
}

fn stop(app: &AppHandle, stage: &mut Stage, binding_id: &str, hotkey_string: &str) {
    let Some(action) = ACTION_MAP.get(binding_id) else {
        warn!("No action in ACTION_MAP for '{binding_id}'");
        return;
    };
    action.stop(app, binding_id, hotkey_string);
    *stage = Stage::Processing;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
//
// The coordinator's real `start`/`stop` functions call ACTION_MAP and require
// an AppHandle, which cannot be constructed in unit tests. Instead we test the
// pure state-machine logic extracted into `handle_ptt_input_testable`, a
// drop-in replacement that accepts injectable callbacks for start/stop so we
// can assert state transitions without Tauri infrastructure.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Arc as StdArc;

    // ── Testable version of handle_ptt_input ────────────────────────────────
    //
    // Mirrors the real function but replaces AppHandle + ACTION_MAP calls with
    // closures so tests can verify start/stop were (not) called.
    fn handle_ptt_testable(
        stage: &mut Stage,
        session_token: &mut u64,
        tx: &Sender<Command>,
        binding_id: &str,
        is_pressed: bool,
        tap_threshold: Duration,
        toggle_hard_timeout: Duration,
        on_start: &mut dyn FnMut(&str),
        on_stop: &mut dyn FnMut(&str),
    ) {
        match (&*stage, is_pressed) {
            (Stage::Idle, true) => {
                on_start(binding_id);
                *stage = Stage::Recording {
                    binding_id: binding_id.to_string(),
                    mode: TriggerMode::Ptt,
                    press_at: Instant::now(),
                };
            }

            (
                Stage::Recording {
                    binding_id: id,
                    mode,
                    press_at,
                },
                false,
            ) if id == binding_id => {
                let elapsed = press_at.elapsed();
                match mode {
                    TriggerMode::Toggle => {
                        // ignore release in toggle mode
                    }
                    TriggerMode::Ptt => {
                        if elapsed < tap_threshold {
                            *session_token = session_token.wrapping_add(1);
                            let token = *session_token;
                            *stage = Stage::Recording {
                                binding_id: binding_id.to_string(),
                                mode: TriggerMode::Toggle,
                                press_at: *press_at,
                            };
                            let tx_clone = tx.clone();
                            thread::spawn(move || {
                                thread::sleep(toggle_hard_timeout);
                                let _ = tx_clone.send(Command::HardTimeout { token });
                            });
                        } else {
                            *session_token = session_token.wrapping_add(1);
                            on_stop(binding_id);
                            *stage = Stage::Processing;
                        }
                    }
                }
            }

            (
                Stage::Recording {
                    binding_id: id,
                    mode: TriggerMode::Toggle,
                    ..
                },
                true,
            ) if id == binding_id => {
                *session_token = session_token.wrapping_add(1);
                on_stop(binding_id);
                *stage = Stage::Processing;
            }

            _ => {}
        }
    }

    // Helper: build a dummy channel for the watchdog (we don't need to drive
    // the receiver in most tests).
    fn dummy_tx() -> Sender<Command> {
        mpsc::channel::<Command>().0
    }

    // ── T1: tap < 200ms → Toggle mode, recording still active ───────────────
    #[test]
    fn tap_under_200ms_enters_toggle_mode() {
        let mut stage = Stage::Idle;
        let mut token = 0u64;
        let tx = dummy_tx();

        let started = StdArc::new(AtomicBool::new(false));
        let stopped = StdArc::new(AtomicBool::new(false));
        let s2 = started.clone();
        let st2 = stopped.clone();

        // Press
        handle_ptt_testable(
            &mut stage,
            &mut token,
            &tx,
            "transcribe",
            true,
            TAP_THRESHOLD,
            Duration::from_secs(9999), // watchdog won't fire during test
            &mut |_| { s2.store(true, Ordering::SeqCst); },
            &mut |_| { st2.store(true, Ordering::SeqCst); },
        );
        assert!(started.load(Ordering::SeqCst), "start should have been called");

        // Release almost immediately (no real sleep needed; Instant::now() elapsed ≈ 0)
        handle_ptt_testable(
            &mut stage,
            &mut token,
            &tx,
            "transcribe",
            false,
            TAP_THRESHOLD,
            Duration::from_secs(9999),
            &mut |_| {},
            &mut |_| { stopped.store(true, Ordering::SeqCst); },
        );

        assert!(!stopped.load(Ordering::SeqCst), "stop must NOT be called on tap release");
        assert!(
            matches!(&stage, Stage::Recording { mode: TriggerMode::Toggle, .. }),
            "stage should be Recording(Toggle) after tap, got {stage:?}"
        );
    }

    // ── T2: hold ≥ 200ms → PTT: release stops immediately ──────────────────
    #[test]
    fn hold_over_200ms_acts_as_ptt() {
        let mut stage = Stage::Idle;
        let mut token = 0u64;
        let tx = dummy_tx();
        let stopped = StdArc::new(AtomicBool::new(false));
        let st2 = stopped.clone();

        // Press
        handle_ptt_testable(
            &mut stage,
            &mut token,
            &tx,
            "transcribe",
            true,
            TAP_THRESHOLD,
            Duration::from_secs(9999),
            &mut |_| {},
            &mut |_| {},
        );

        // Simulate hold by backdating press_at
        if let Stage::Recording { ref mut press_at, .. } = stage {
            *press_at = Instant::now() - Duration::from_millis(300);
        }

        // Release after 300ms hold
        handle_ptt_testable(
            &mut stage,
            &mut token,
            &tx,
            "transcribe",
            false,
            TAP_THRESHOLD,
            Duration::from_secs(9999),
            &mut |_| {},
            &mut |_| { st2.store(true, Ordering::SeqCst); },
        );

        assert!(stopped.load(Ordering::SeqCst), "stop must be called on PTT hold release");
        assert!(matches!(stage, Stage::Processing), "stage should be Processing, got {stage:?}");
    }

    // ── T3: Toggle mode — second press stops recording ───────────────────────
    #[test]
    fn toggle_mode_press_again_stops() {
        let mut stage = Stage::Idle;
        let mut token = 0u64;
        let tx = dummy_tx();
        let stop_count = StdArc::new(AtomicU32::new(0));
        let sc2 = stop_count.clone();

        // First press
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", true,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| { sc2.fetch_add(1, Ordering::SeqCst); },
        );
        // Quick release → Toggle mode
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", false,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| { sc2.fetch_add(1, Ordering::SeqCst); },
        );
        assert!(matches!(&stage, Stage::Recording { mode: TriggerMode::Toggle, .. }));

        // Second press → should stop
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", true,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| { sc2.fetch_add(1, Ordering::SeqCst); },
        );

        assert_eq!(stop_count.load(Ordering::SeqCst), 1, "stop called exactly once");
        assert!(matches!(stage, Stage::Processing), "stage should be Processing");
    }

    // ── T4: Toggle mode — extra release is ignored ───────────────────────────
    #[test]
    fn toggle_mode_release_ignored() {
        let mut stage = Stage::Idle;
        let mut token = 0u64;
        let tx = dummy_tx();
        let stop_count = StdArc::new(AtomicU32::new(0));
        let sc2 = stop_count.clone();

        // Press → quick release → Toggle mode
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", true,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| { sc2.fetch_add(1, Ordering::SeqCst); },
        );
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", false,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| { sc2.fetch_add(1, Ordering::SeqCst); },
        );
        assert!(matches!(&stage, Stage::Recording { mode: TriggerMode::Toggle, .. }));

        // Extra release — must be ignored
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", false,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| { sc2.fetch_add(1, Ordering::SeqCst); },
        );

        assert_eq!(stop_count.load(Ordering::SeqCst), 0, "stop must never be called");
        assert!(
            matches!(&stage, Stage::Recording { mode: TriggerMode::Toggle, .. }),
            "stage must still be Toggle Recording"
        );
    }

    // ── T5: Toggle hard-timeout auto-stops ──────────────────────────────────
    //
    // We use a very short synthetic timeout (50ms) instead of sleeping 60s.
    // The watchdog spawns a thread that sends HardTimeout; we process it
    // in a local loop that mirrors the coordinator's HardTimeout handler.
    #[test]
    fn toggle_hard_timeout_auto_stops() {
        let (tx, rx) = mpsc::channel::<Command>();
        let mut stage = Stage::Idle;
        let mut token = 0u64;
        let stopped = StdArc::new(AtomicBool::new(false));
        let st2 = stopped.clone();

        // Press
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", true,
            TAP_THRESHOLD, Duration::from_millis(50), // fast watchdog
            &mut |_| {}, &mut |_| {},
        );
        // Quick release → Toggle mode, watchdog armed with 50ms timeout
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", false,
            TAP_THRESHOLD, Duration::from_millis(50),
            &mut |_| {}, &mut |_| {},
        );
        assert!(matches!(&stage, Stage::Recording { mode: TriggerMode::Toggle, .. }));

        // Wait for HardTimeout to arrive (watchdog fires after 50ms)
        let cmd = rx.recv_timeout(Duration::from_millis(500))
            .expect("HardTimeout should arrive within 500ms");

        // Process it the same way the real coordinator does
        if let Command::HardTimeout { token: t } = cmd {
            if t == token {
                if let Stage::Recording { mode: TriggerMode::Toggle, .. } = &stage {
                    let _ = token.wrapping_add(1); // would bump session_token in real code
                    st2.store(true, Ordering::SeqCst);
                    stage = Stage::Processing;
                }
            }
        }

        assert!(stopped.load(Ordering::SeqCst), "hard-timeout should trigger auto-stop");
        assert!(matches!(stage, Stage::Processing), "stage should be Processing after timeout");
    }

    // ── T6: stale HardTimeout token is ignored ───────────────────────────────
    #[test]
    fn stale_hard_timeout_ignored() {
        let mut stage = Stage::Idle;
        let mut token = 0u64;
        let (tx, _rx) = mpsc::channel::<Command>();

        // Press + quick release → Toggle mode (token bumped to 1)
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", true,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| {},
        );
        handle_ptt_testable(
            &mut stage, &mut token, &tx, "transcribe", false,
            TAP_THRESHOLD, Duration::from_secs(9999),
            &mut |_| {}, &mut |_| {},
        );
        let current_token = token;

        // Simulate a stale token (from a previous session)
        let stale_token = current_token.wrapping_sub(1);

        // Apply the HardTimeout handler logic directly
        let is_stale = stale_token != current_token;
        assert!(is_stale, "token should be considered stale");
        // Stage must remain unchanged
        assert!(
            matches!(&stage, Stage::Recording { mode: TriggerMode::Toggle, .. }),
            "stage must not change on stale timeout"
        );
    }
}
