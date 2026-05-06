use crate::audio_toolkit::{list_input_devices, vad::SmoothedVad, AudioRecorder, SileroVad};
use crate::audio_toolkit::audio::chunker::{AudioChunker, ChunkerConfig};

/// FD-006 M2: type alias for the incremental-paste callback passed to the drainer.
type AppendPasteCb = Arc<Mutex<Box<dyn Fn(&str) + Send + 'static>>>;
use crate::helpers::clamshell;
use crate::observability::{self, Outcome, Stage, Stopwatch};
use crate::output::{select_sink_auto, DeltaComputer, StreamingSink};
use crate::settings::{get_settings, AppSettings};
use crate::streaming_pipeline::{ChunkPartial, OrchestratorConfig, StreamingOrchestrator};
use crate::utils;
use log::{debug, error, info, warn};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::Manager;

fn set_mute(mute: bool) {
    // Expected behavior:
    // - Windows: works on most systems using standard audio drivers.
    // - Linux: works on many systems (PipeWire, PulseAudio, ALSA),
    //   but some distros may lack the tools used.
    // - macOS: works on most standard setups via AppleScript.
    // If unsupported, fails silently.

    #[cfg(target_os = "windows")]
    {
        unsafe {
            use windows::Win32::{
                Media::Audio::{
                    eMultimedia, eRender, Endpoints::IAudioEndpointVolume, IMMDeviceEnumerator,
                    MMDeviceEnumerator,
                },
                System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED},
            };

            macro_rules! unwrap_or_return {
                ($expr:expr) => {
                    match $expr {
                        Ok(val) => val,
                        Err(_) => return,
                    }
                };
            }

            // Initialize the COM library for this thread.
            // If already initialized (e.g., by another library like Tauri), this does nothing.
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

            let all_devices: IMMDeviceEnumerator =
                unwrap_or_return!(CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL));
            let default_device =
                unwrap_or_return!(all_devices.GetDefaultAudioEndpoint(eRender, eMultimedia));
            let volume_interface = unwrap_or_return!(
                default_device.Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None)
            );

            let _ = volume_interface.SetMute(mute, std::ptr::null());
        }
    }

    #[cfg(target_os = "linux")]
    {
        use std::process::Command;

        let mute_val = if mute { "1" } else { "0" };
        let amixer_state = if mute { "mute" } else { "unmute" };

        // Try multiple backends to increase compatibility
        // 1. PipeWire (wpctl)
        if Command::new("wpctl")
            .args(["set-mute", "@DEFAULT_AUDIO_SINK@", mute_val])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return;
        }

        // 2. PulseAudio (pactl)
        if Command::new("pactl")
            .args(["set-sink-mute", "@DEFAULT_SINK@", mute_val])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return;
        }

        // 3. ALSA (amixer)
        let _ = Command::new("amixer")
            .args(["set", "Master", amixer_state])
            .output();
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let script = format!(
            "set volume output muted {}",
            if mute { "true" } else { "false" }
        );
        let _ = Command::new("osascript").args(["-e", &script]).output();
    }
}

const WHISPER_SAMPLE_RATE: usize = 16000;

/* ──────────────────────────────────────────────────────────────── */

#[derive(Clone, Debug)]
pub enum RecordingState {
    Idle,
    Recording { binding_id: String },
}

#[derive(Clone, Debug)]
pub enum MicrophoneMode {
    AlwaysOn,
    OnDemand,
}

/* ──────────────────────────────────────────────────────────────── */

fn create_audio_recorder(
    vad_path: &str,
    app_handle: &tauri::AppHandle,
) -> Result<AudioRecorder, anyhow::Error> {
    // FD-003 M3.5 #3: threshold 0.3→0.6 (reduce noise mis-triggers),
    // prefill 15→8 frames (reduce first-word truncation; 240ms pre-roll is
    // enough for SenseVoice), hangover 15→4 frames (reduce tail-silence wait
    // from 450ms to ~120ms; audit target min_silence_duration_ms 250→100).
    let silero = SileroVad::new(vad_path, 0.6)
        .map_err(|e| anyhow::anyhow!("Failed to create SileroVad: {}", e))?;
    let smoothed_vad = SmoothedVad::new(Box::new(silero), 8, 4, 2);

    // Recorder with VAD plus a spectrum-level callback that forwards updates to
    // the frontend.
    let recorder = AudioRecorder::new()
        .map_err(|e| anyhow::anyhow!("Failed to create AudioRecorder: {}", e))?
        .with_vad(Box::new(smoothed_vad))
        .with_level_callback({
            let app_handle = app_handle.clone();
            move |levels| {
                utils::emit_levels(&app_handle, &levels);
            }
        });

    Ok(recorder)
}

/* ──────────────────────────────────────────────────────────────── */

/// FD-006 M2: shared state for a single chunked-streaming session.
///
/// Created by `enable_streaming()` when a qwen3_mlx recording starts and torn
/// down by `disable_streaming()` when the recording ends (or is cancelled).
struct StreamingSession {
    orchestrator: Arc<StreamingOrchestrator>,
    drainer_handle: Option<std::thread::JoinHandle<()>>,
    /// Signals the drainer thread to stop.
    drainer_stop: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Clone)]
pub struct AudioRecordingManager {
    state: Arc<Mutex<RecordingState>>,
    mode: Arc<Mutex<MicrophoneMode>>,
    app_handle: tauri::AppHandle,

    recorder: Arc<Mutex<Option<AudioRecorder>>>,
    is_open: Arc<Mutex<bool>>,
    is_recording: Arc<Mutex<bool>>,
    did_mute: Arc<Mutex<bool>>,

    /// FD-006 M2: active chunked-streaming session (Some while recording with
    /// qwen3_mlx_streaming_chunked = true, None otherwise).
    streaming_session: Arc<Mutex<Option<StreamingSession>>>,
}

impl AudioRecordingManager {
    /* ---------- construction ------------------------------------------------ */

    pub fn new(app: &tauri::AppHandle) -> Result<Self, anyhow::Error> {
        let settings = get_settings(app);
        let mode = if settings.always_on_microphone {
            MicrophoneMode::AlwaysOn
        } else {
            MicrophoneMode::OnDemand
        };

        let manager = Self {
            state: Arc::new(Mutex::new(RecordingState::Idle)),
            mode: Arc::new(Mutex::new(mode.clone())),
            app_handle: app.clone(),

            recorder: Arc::new(Mutex::new(None)),
            is_open: Arc::new(Mutex::new(false)),
            is_recording: Arc::new(Mutex::new(false)),
            did_mute: Arc::new(Mutex::new(false)),

            streaming_session: Arc::new(Mutex::new(None)),
        };

        // Always-on?  Open immediately.
        if matches!(mode, MicrophoneMode::AlwaysOn) {
            manager.start_microphone_stream()?;
        }

        Ok(manager)
    }

    /* ---------- helper methods --------------------------------------------- */

    fn get_effective_microphone_device(&self, settings: &AppSettings) -> Option<cpal::Device> {
        // Check if we're in clamshell mode and have a clamshell microphone configured
        let use_clamshell_mic = if let Ok(is_clamshell) = clamshell::is_clamshell() {
            is_clamshell && settings.clamshell_microphone.is_some()
        } else {
            false
        };

        let device_name = if use_clamshell_mic {
            settings.clamshell_microphone.as_ref().unwrap()
        } else {
            settings.selected_microphone.as_ref()?
        };

        // Find the device by name
        match list_input_devices() {
            Ok(devices) => devices
                .into_iter()
                .find(|d| d.name == *device_name)
                .map(|d| d.device),
            Err(e) => {
                debug!("Failed to list devices, using default: {}", e);
                None
            }
        }
    }

    /* ---------- microphone life-cycle -------------------------------------- */

    /// Applies mute if mute_while_recording is enabled and stream is open
    pub fn apply_mute(&self) {
        let settings = get_settings(&self.app_handle);
        let mut did_mute_guard = self.did_mute.lock().unwrap();

        if settings.mute_while_recording && *self.is_open.lock().unwrap() {
            set_mute(true);
            *did_mute_guard = true;
            debug!("Mute applied");
        }
    }

    /// Removes mute if it was applied
    pub fn remove_mute(&self) {
        let mut did_mute_guard = self.did_mute.lock().unwrap();
        if *did_mute_guard {
            set_mute(false);
            *did_mute_guard = false;
            debug!("Mute removed");
        }
    }

    pub fn preload_vad(&self) -> Result<(), anyhow::Error> {
        let mut recorder_opt = self.recorder.lock().unwrap();
        if recorder_opt.is_none() {
            let vad_path = self
                .app_handle
                .path()
                .resolve(
                    "resources/models/silero_vad_v4.onnx",
                    tauri::path::BaseDirectory::Resource,
                )
                .map_err(|e| anyhow::anyhow!("Failed to resolve VAD path: {}", e))?;
            *recorder_opt = Some(create_audio_recorder(
                vad_path.to_str().unwrap(),
                &self.app_handle,
            )?);
        }
        Ok(())
    }

    pub fn start_microphone_stream(&self) -> Result<(), anyhow::Error> {
        let mut open_flag = self.is_open.lock().unwrap();
        if *open_flag {
            debug!("Microphone stream already active");
            return Ok(());
        }

        let start_time = Instant::now();
        let t1_sw = Stopwatch::start();

        // Don't mute immediately - caller will handle muting after audio feedback
        let mut did_mute_guard = self.did_mute.lock().unwrap();
        *did_mute_guard = false;

        // Get the selected device from settings, considering clamshell mode
        let settings = get_settings(&self.app_handle);
        let selected_device = self.get_effective_microphone_device(&settings);

        // Pre-flight check: if no device was selected/configured AND no devices
        // exist at all, fail early with a clear error instead of letting cpal
        // produce a cryptic backend-specific message.
        if selected_device.is_none() {
            let has_any_device = list_input_devices()
                .map(|devices| !devices.is_empty())
                .unwrap_or(false);
            if !has_any_device {
                return Err(anyhow::anyhow!("No input device found"));
            }
        }

        // Ensure VAD is loaded if it wasn't for whatever reason
        self.preload_vad()?;

        let mut recorder_opt = self.recorder.lock().unwrap();
        if let Some(rec) = recorder_opt.as_mut() {
            if let Err(e) = rec.open(selected_device) {
                let req = self
                    .app_handle
                    .try_state::<crate::observability::ActiveRequestId>()
                    .map(|s| s.get())
                    .unwrap_or_default();
                observability::record_stage(
                    req,
                    Stage::T1AudioCapture,
                    Outcome::Error,
                    t1_sw.elapsed_ms(),
                    Some(serde_json::json!({ "error": e.to_string() })),
                );
                return Err(anyhow::anyhow!("Failed to open recorder: {}", e));
            }
        }

        *open_flag = true;
        // This timing covers through cpal's stream.play() returning — i.e. the
        // point cpal surfaces as "stream running." It does NOT guarantee the
        // host audio device is producing samples yet; the first input callback
        // fires asynchronously one buffer period later (hardware dependent,
        // typically ~10–200ms on macOS, longer on Bluetooth/USB).
        let capture_open_ms = t1_sw.elapsed_ms();
        info!(
            "Microphone stream initialized in {:?}",
            start_time.elapsed()
        );

        // t1_audio_capture: cpal stream.play() returned
        // We have no active request_id here (audio manager has no app state ref
        // for the current request), so we use a sentinel.  The active req is
        // picked up in actions.rs from the Tauri state.
        let req = self
            .app_handle
            .try_state::<crate::observability::ActiveRequestId>()
            .map(|s| s.get())
            .unwrap_or_default();
        observability::ok_with(
            req,
            Stage::T1AudioCapture,
            capture_open_ms,
            serde_json::json!({ "capture_open_ms": capture_open_ms as u64 }),
        );

        Ok(())
    }

    pub fn stop_microphone_stream(&self) {
        let mut open_flag = self.is_open.lock().unwrap();
        if !*open_flag {
            return;
        }

        let mut did_mute_guard = self.did_mute.lock().unwrap();
        if *did_mute_guard {
            set_mute(false);
        }
        *did_mute_guard = false;

        if let Some(rec) = self.recorder.lock().unwrap().as_mut() {
            // If still recording, stop first.
            if *self.is_recording.lock().unwrap() {
                let _ = rec.stop();
                *self.is_recording.lock().unwrap() = false;
            }
            let _ = rec.close();
        }

        *open_flag = false;
        debug!("Microphone stream stopped");
    }

    /* ---------- mode switching --------------------------------------------- */

    pub fn update_mode(&self, new_mode: MicrophoneMode) -> Result<(), anyhow::Error> {
        let cur_mode = self.mode.lock().unwrap().clone();

        match (cur_mode, &new_mode) {
            (MicrophoneMode::AlwaysOn, MicrophoneMode::OnDemand) => {
                if matches!(*self.state.lock().unwrap(), RecordingState::Idle) {
                    self.stop_microphone_stream();
                }
            }
            (MicrophoneMode::OnDemand, MicrophoneMode::AlwaysOn) => {
                self.start_microphone_stream()?;
            }
            _ => {}
        }

        *self.mode.lock().unwrap() = new_mode;
        Ok(())
    }

    /* ---------- recording --------------------------------------------------- */

    pub fn try_start_recording(&self, binding_id: &str) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();

        if let RecordingState::Idle = *state {
            // Ensure microphone is open in on-demand mode
            if matches!(*self.mode.lock().unwrap(), MicrophoneMode::OnDemand) {
                if let Err(e) = self.start_microphone_stream() {
                    let msg = format!("{e}");
                    error!("Failed to open microphone stream: {msg}");
                    return Err(msg);
                }
            }

            if let Some(rec) = self.recorder.lock().unwrap().as_ref() {
                if rec.start().is_ok() {
                    *self.is_recording.lock().unwrap() = true;
                    *state = RecordingState::Recording {
                        binding_id: binding_id.to_string(),
                    };
                    debug!("Recording started for binding {binding_id}");
                    return Ok(());
                }
            }
            Err("Recorder not available".to_string())
        } else {
            Err("Already recording".to_string())
        }
    }

    pub fn update_selected_device(&self) -> Result<(), anyhow::Error> {
        // If currently open, restart the microphone stream to use the new device
        if *self.is_open.lock().unwrap() {
            self.stop_microphone_stream();
            self.start_microphone_stream()?;
        }
        Ok(())
    }

    pub fn stop_recording(&self, binding_id: &str) -> Option<Vec<f32>> {
        let mut state = self.state.lock().unwrap();

        match *state {
            RecordingState::Recording {
                binding_id: ref active,
            } if active == binding_id => {
                *state = RecordingState::Idle;
                drop(state);

                // Optionally keep recording for a bit longer to capture trailing audio
                let settings = get_settings(&self.app_handle);
                if settings.extra_recording_buffer_ms > 0 {
                    debug!(
                        "Extra recording buffer: sleeping {}ms before stopping",
                        settings.extra_recording_buffer_ms
                    );
                    std::thread::sleep(Duration::from_millis(settings.extra_recording_buffer_ms));
                }

                let samples = if let Some(rec) = self.recorder.lock().unwrap().as_ref() {
                    match rec.stop() {
                        Ok(buf) => buf,
                        Err(e) => {
                            error!("stop() failed: {e}");
                            Vec::new()
                        }
                    }
                } else {
                    error!("Recorder not available");
                    Vec::new()
                };

                *self.is_recording.lock().unwrap() = false;

                // In on-demand mode, close the mic immediately after recording.
                if matches!(*self.mode.lock().unwrap(), MicrophoneMode::OnDemand) {
                    self.stop_microphone_stream();
                }

                // Pad if very short
                let s_len = samples.len();
                // debug!("Got {} samples", s_len);
                if s_len < WHISPER_SAMPLE_RATE && s_len > 0 {
                    let mut padded = samples;
                    padded.resize(WHISPER_SAMPLE_RATE * 5 / 4, 0.0);
                    Some(padded)
                } else {
                    Some(samples)
                }
            }
            _ => None,
        }
    }
    pub fn is_recording(&self) -> bool {
        matches!(
            *self.state.lock().unwrap(),
            RecordingState::Recording { .. }
        )
    }

    /// Cancel any ongoing recording without returning audio samples
    pub fn cancel_recording(&self) {
        let mut state = self.state.lock().unwrap();

        if let RecordingState::Recording { .. } = *state {
            *state = RecordingState::Idle;
            drop(state);

            if let Some(rec) = self.recorder.lock().unwrap().as_ref() {
                let _ = rec.stop(); // Discard the result
            }

            *self.is_recording.lock().unwrap() = false;

            // FD-006 M2: cancel any active streaming session.
            self.cancel_streaming();

            // In on-demand mode, close the mic immediately after cancelling.
            if matches!(*self.mode.lock().unwrap(), MicrophoneMode::OnDemand) {
                self.stop_microphone_stream();
            }
        }
    }

    // ── FD-006 M2: chunked streaming helpers ──────────────────────────────────

    /// Enable chunked streaming for the current recording session.
    ///
    /// Creates a `StreamingOrchestrator` worker and a drainer thread that
    /// continuously pulls `ChunkPartial`s → `DeltaComputer` → `sink.append`.
    ///
    /// Must be called **before** `try_start_recording` so the chunker is wired
    /// before the first audio frame arrives.
    ///
    /// `append_paste` is a closure supplied by `TranscriptionManager`; it calls
    /// `self.append_incremental_paste(delta)` without creating a circular dep
    /// from audio.rs → transcription.rs.
    pub fn start_chunked_streaming<F>(&self, append_paste: F)
    where
        F: Fn(&str) + Send + 'static,
    {
        use crate::streaming_pipeline::orchestrator::MlxAudioBackend;
        use std::sync::atomic::Ordering as AOrdering;

        // Tear down any previous session.
        self.stop_chunked_streaming(false);

        // ── Orchestrator ──────────────────────────────────────────────────
        let backend = Box::new(MlxAudioBackend);
        let orch_config = OrchestratorConfig::default();
        let orch = Arc::new(StreamingOrchestrator::start(orch_config, backend));

        // ── Chunker + per-frame callback ──────────────────────────────────
        let chunker_arc = Arc::new(Mutex::new(AudioChunker::new(ChunkerConfig::default())));
        let orch_for_cb = Arc::clone(&orch);
        let chunker_for_cb = Arc::clone(&chunker_arc);

        {
            let mut rec_guard = self.recorder.lock().unwrap();
            if let Some(rec) = rec_guard.as_mut() {
                rec.set_streaming_frame_callback(Some(Arc::new(
                    move |frame: &[f32], is_speech: bool| {
                        let maybe_chunk = {
                            let mut ch = chunker_for_cb.lock().unwrap();
                            ch.ingest(frame, is_speech)
                        };
                        if let Some(chunk) = maybe_chunk {
                            if let Err(e) = orch_for_cb.submit(chunk) {
                                // submit returns Err on cancel/shutdown — not
                                // a bug, just a race at session end.
                                debug!("[chunker] orchestrator submit: {}", e);
                            }
                        }
                    },
                )));
            } else {
                warn!("[streaming] Recorder not available; streaming disabled");
                return;
            }
        }

        // ── Drainer thread ────────────────────────────────────────────────
        // Polls the partial channel at ~50 ms cadence.  On drainer_stop signal,
        // drains remaining partials then finalizes the sink and exits.
        let drainer_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let drainer_stop_clone = Arc::clone(&drainer_stop);
        let orch_for_drainer = Arc::clone(&orch);

        // Wrap the append callback behind a Mutex so it can cross thread boundary.
        let append_paste_arc: AppendPasteCb =
            Arc::new(Mutex::new(Box::new(append_paste)));

        let drainer_handle = std::thread::spawn(move || {
            crate::platform::elevate_thread_qos("streaming-drainer");
            let mut delta_computer = DeltaComputer::new();
            let mut sink: Box<dyn StreamingSink + Send> = select_sink_auto();
            let timeout = Duration::from_millis(50);
            // Track when the last partial arrived to compute chunk_emit_interval_ms.
            let mut last_partial_at: Option<Instant> = None;

            loop {
                if drainer_stop_clone.load(AOrdering::Relaxed) {
                    // Drain all remaining partials before exit.
                    while let Some(partial) = orch_for_drainer.try_recv_partial() {
                        let interval_ms = emit_interval_ms(&mut last_partial_at);
                        process_chunk_partial(
                            &partial,
                            &mut delta_computer,
                            sink.as_mut(),
                            &append_paste_arc,
                            interval_ms,
                        );
                    }
                    // Finalize sink.
                    if let Err(e) = sink.finalize() {
                        warn!("[streaming-drainer] sink.finalize failed: {}", e);
                    }
                    break;
                }

                if let Some(partial) = orch_for_drainer.recv_partial_timeout(timeout) {
                    let interval_ms = emit_interval_ms(&mut last_partial_at);
                    process_chunk_partial(
                        &partial,
                        &mut delta_computer,
                        sink.as_mut(),
                        &append_paste_arc,
                        interval_ms,
                    );
                }
            }
        });

        *self.streaming_session.lock().unwrap() = Some(StreamingSession {
            orchestrator: orch,
            drainer_handle: Some(drainer_handle),
            drainer_stop,
        });

        info!("[streaming] Chunked streaming session started");
    }

    /// Gracefully stop the chunked streaming session after recording ends.
    ///
    /// `do_finalize = true` → drainer drains remaining partials + finalizes sink.
    /// `do_finalize = false` → cancel orchestrator (skip pending) + stop drainer.
    pub fn stop_chunked_streaming(&self, do_finalize: bool) {
        // Remove the streaming frame callback so no more chunks are submitted.
        {
            let mut rec_guard = self.recorder.lock().unwrap();
            if let Some(rec) = rec_guard.as_mut() {
                rec.set_streaming_frame_callback(None);
            }
        }

        let session_opt = self.streaming_session.lock().unwrap().take();
        if let Some(mut session) = session_opt {
            if !do_finalize {
                session.orchestrator.cancel();
            }
            // Signal drainer to drain+exit.
            session
                .drainer_stop
                .store(true, std::sync::atomic::Ordering::SeqCst);
            if let Some(handle) = session.drainer_handle.take() {
                // Give the drainer up to 3 s to finish then detach.
                let _ = handle.join();
            }
            info!(
                "[streaming] Chunked streaming session stopped (finalize={})",
                do_finalize
            );
        }
    }

    /// Cancel the streaming session — orchestrator cancelled, drainer stops
    /// without emitting remaining partials.
    pub fn cancel_streaming(&self) {
        self.stop_chunked_streaming(false);
    }

    /// Whether a chunked streaming session is currently active.
    pub fn is_streaming_active(&self) -> bool {
        self.streaming_session.lock().unwrap().is_some()
    }
}

// ── FD-006 M2: drainer helper ─────────────────────────────────────────────────

/// Compute the elapsed milliseconds since the last partial arrived, updating
/// `last_at`.  Returns `None` for the very first partial (no prior baseline).
fn emit_interval_ms(last_at: &mut Option<Instant>) -> Option<f64> {
    let now = Instant::now();
    let interval = last_at.map(|prev| now.duration_since(prev).as_secs_f64() * 1000.0);
    *last_at = Some(now);
    interval
}

/// Process one `ChunkPartial` from the orchestrator:
///   partial_text → DeltaComputer → Action::Append(delta) → sink.append → callback.
///
/// `chunk_emit_interval_ms`: elapsed since the previous partial was processed
/// (None for the first partial in a session).  Emitted to observability.
fn process_chunk_partial(
    partial: &ChunkPartial,
    _delta_computer: &mut DeltaComputer,
    sink: &mut (dyn StreamingSink + Send),
    append_paste: &AppendPasteCb,
    chunk_emit_interval_ms: Option<f64>,
) {
    if partial.partial_text.is_empty() {
        return;
    }
    if let Some(ref err) = partial.error {
        warn!(
            "[streaming-drainer] chunk_idx={} inference error: {}",
            partial.chunk_idx, err
        );
        return;
    }

    // Emit T5aChunkInference span with chunk_emit_interval_ms so the SLA
    // assert on "speech → partial on screen" latency has data.
    if let Some(interval) = chunk_emit_interval_ms {
        let req = crate::observability::RequestId::new();
        crate::observability::ok_with(
            req,
            crate::observability::Stage::T5aChunkInference,
            partial.inference_ms,
            serde_json::json!({
                crate::observability::OBS_FIELD_CHUNK_IDX: partial.chunk_idx,
                crate::observability::OBS_FIELD_CHUNK_EMIT_INTERVAL_MS: interval,
                "inference_ms": partial.inference_ms,
            }),
        );
    }

    // FD-006 M2 follow-up #2: each chunk's partial is the transcription of a
    // ~1s audio window — NOT a cumulative growing partial like FD-003 M2.6's
    // pseudo-streaming Qwen3 token callback. Adjacent chunk partials therefore
    // do not have a prefix relationship (chunk N = "切换一下", chunk N+1 =
    // "一下我们"). The conservative DeltaComputer.compute would Skip every
    // non-first chunk because starts_with(prev) is false — symptom user
    // reported: only first sentence appears, rest "stuck".
    //
    // In chunked mode we bypass DeltaComputer and append each chunk's text
    // directly to the sink + incremental_paste_cursor. The 300 ms overlap +
    // model boundary errors are tolerated during streaming; M3 final pass
    // batch-inferences the full audio and reconciles via backspace + retype
    // when it ships, replacing the streamed approximation with the
    // authoritative transcript.
    let delta = partial.partial_text.as_str();
    let chars_appended = delta.chars().count();
    match sink.append(delta) {
        Ok(()) => {
            if let Ok(cb) = append_paste.lock() {
                cb(delta);
            }
            info!(
                "[t7_output] streaming=true chunked=true chunk_idx={} delta_chars={}",
                partial.chunk_idx, chars_appended
            );
        }
        Err(e) => {
            warn!("[streaming-drainer] sink.append failed: {}", e);
        }
    }
}
