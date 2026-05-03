use crate::audio_toolkit::{apply_custom_words, filter_transcription_output, VoiceActivityDetector};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::{EngineType, ModelManager, SherpaModelKind};
use crate::profile_resolver::resolve_effective_settings;
use crate::settings::{
    get_settings, ModelUnloadTimeout, OrtAcceleratorSetting, WhisperAcceleratorSetting,
};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::Serialize;
use sherpa_onnx::{
    OfflineFunASRNanoModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer,
    OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
};
use specta::Type;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use transcribe_rs::{
    onnx::{
        canary::CanaryModel,
        cohere::CohereModel,
        gigaam::GigaAMModel,
        moonshine::{MoonshineModel, MoonshineVariant, StreamingModel},
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    whisper_cpp::{WhisperEngine, WhisperInferenceParams},
    SpeechModel, TranscribeOptions,
};

/// Holds a loaded sherpa-onnx offline recognizer along with its model kind so the
/// transcribe dispatch knows which recognition path to use.
struct SherpaSession {
    recognizer: OfflineRecognizer,
    /// Model family — used for dispatch (e.g. Qwen3Asr VAD chunking path).
    kind: SherpaModelKind,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

enum LoadedEngine {
    Whisper(WhisperEngine),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Canary(CanaryModel),
    Cohere(CohereModel),
    /// Apple Speech (SFSpeechRecognizer) — no persistent model state; each
    /// call creates a new SFSpeechAudioBufferRecognitionRequest internally.
    /// The _default_locale field stores the BCP-47 locale resolved at load time
    /// (reserved for future use when streaming partial results are added).
    AppleSpeech {
        _default_locale: String,
    },
    /// sherpa-onnx offline recognizer (k2-fsa upstream crate).
    /// Supports SenseVoice, FunASR-Nano, and Qwen3-ASR model families.
    Sherpa(SherpaSession),
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .map_or(false, |a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        let engine = self.lock_engine();
        engine.is_some()
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Dropping the engine frees all resources
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            if let Err(e) = self.unload_model() {
                warn!("Failed to immediately unload model: {}", e);
            }
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        // AppleSpeech is a virtual model with no on-disk file.
        // Skip get_model_path for it and short-circuit early.
        #[cfg(target_os = "macos")]
        if matches!(model_info.engine_type, EngineType::AppleSpeech) {
            let emit_loading_failed_apple = |error_msg: &str| {
                let _ = self.app_handle.emit(
                    "model-state-changed",
                    ModelStateEvent {
                        event_type: "loading_failed".to_string(),
                        model_id: Some(model_id.to_string()),
                        model_name: Some(model_info.name.clone()),
                        error: Some(error_msg.to_string()),
                    },
                );
            };
            if !crate::apple_speech::is_apple_speech_available() {
                let error_msg = "Apple Speech is not available on this device";
                emit_loading_failed_apple(error_msg);
                return Err(anyhow::anyhow!(error_msg));
            }
            info!("Apple Speech engine ready (no model to load)");
            // Resolve the default locale from the user's selected_language setting
            // so future consumers (e.g. partial-result streaming) don't fall back
            // to en-US on a Chinese / Japanese / etc. system.
            let resolved_locale = map_to_bcp47(&get_settings(&self.app_handle).selected_language);
            let loaded_engine = LoadedEngine::AppleSpeech {
                _default_locale: resolved_locale,
            };
            {
                let mut engine = self.lock_engine();
                *engine = Some(loaded_engine);
            }
            {
                let mut current_model = self.current_model_id.lock().unwrap();
                *current_model = Some(model_id.to_string());
            }
            self.touch_activity();
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_completed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: None,
                },
            );
            let load_duration = load_start.elapsed();
            debug!(
                "Apple Speech engine loaded (took {}ms)",
                load_duration.as_millis()
            );
            return Ok(());
        }

        let model_path = self.model_manager.get_model_path(model_id)?;

        // Create appropriate engine based on model type
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        let loaded_engine = match model_info.engine_type {
            EngineType::Whisper => {
                let engine = WhisperEngine::load(&model_path).map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Whisper(engine)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 0, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let engine = GigaAMModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load gigaam model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::GigaAM(engine)
            }
            EngineType::Canary => {
                let engine = CanaryModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load canary model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Canary(engine)
            }
            EngineType::Cohere => {
                let engine = CohereModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load cohere model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Cohere(engine)
            }
            EngineType::AppleSpeech => {
                // On macOS this branch is unreachable: AppleSpeech is handled
                // by the early-return block above (before get_model_path).
                // On other platforms we still need a match arm for exhaustiveness.
                let error_msg = "Apple Speech is only available on macOS";
                emit_loading_failed(error_msg);
                return Err(anyhow::anyhow!(error_msg));
            }
            EngineType::Sherpa(kind) => {
                // Build the OfflineRecognizerConfig appropriate for each model family.
                // CPU + ONNX Runtime is the right default on Apple Silicon — sherpa-onnx
                // issue #2910 shows the CoreML execution provider regresses RTF for
                // Encoder+LLM models like FunASR-Nano (KV-cache fallback overhead),
                // so we explicitly stay on CPU. Probe physical parallelism and clamp
                // to [4, 6] — upper bound stays below typical M-series efficiency-core
                // crossover so onnx threads don't get scheduled onto E-cores.
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .clamp(4, 6) as i32;
                let mut config = OfflineRecognizerConfig::default();
                config.model_config.num_threads = num_threads;
                config.model_config.provider = Some("cpu".to_string());

                match &kind {
                    SherpaModelKind::SenseVoice => {
                        // model.int8.onnx + tokens.txt live at the top level of the
                        // extracted directory.
                        config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
                            model: Some(
                                model_path
                                    .join("model.int8.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            // Language is set per-transcription (accept "auto" → None).
                            language: Some("auto".into()),
                            use_itn: true,
                        };
                        config.model_config.tokens =
                            Some(model_path.join("tokens.txt").to_string_lossy().into_owned());

                        // Inject hotwords from settings.custom_words (L2 bias).
                        // sherpa-onnx requires the model_config.modeling_unit +
                        // bpe_vocab to be set so it can tokenize hotwords against
                        // the model's actual vocabulary; without this the file is
                        // written but contributes ~no bias for CJK or English BPE
                        // tokens. SenseVoice ships a SentencePiece BPE tokens.txt
                        // (rows like `▁the`), which doubles as the bpe_vocab.
                        let settings = get_settings(&self.app_handle);
                        if !settings.custom_words.is_empty() {
                            match crate::portable::app_data_dir(&self.app_handle) {
                                Ok(data_dir) => {
                                    let hw_path = data_dir.join("sense_voice_hotwords.txt");
                                    // One word per line; sherpa-onnx tokenizes
                                    // each line via modeling_unit + bpe_vocab.
                                    let contents = settings
                                        .custom_words
                                        .iter()
                                        .map(|w| w.trim())
                                        .filter(|w| !w.is_empty())
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    match std::fs::write(&hw_path, &contents) {
                                        Ok(()) => {
                                            config.hotwords_file =
                                                Some(hw_path.to_string_lossy().into_owned());
                                            config.hotwords_score = settings.hotwords_boost;
                                            // Critical: enable cjkchar+bpe tokenization
                                            // for hotwords. Without it CJK words and
                                            // English BPE pieces can't be biased.
                                            config.model_config.modeling_unit =
                                                Some("cjkchar+bpe".into());
                                            config.model_config.bpe_vocab = Some(
                                                model_path
                                                    .join("tokens.txt")
                                                    .to_string_lossy()
                                                    .into_owned(),
                                            );
                                            debug!(
                                                "SenseVoice sherpa: hotwords_file={:?} score={} modeling_unit=cjkchar+bpe",
                                                hw_path, settings.hotwords_boost
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to write sense_voice_hotwords.txt: {}; hotwords disabled",
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to get app_data_dir for hotwords file: {}; hotwords disabled",
                                        e
                                    );
                                }
                            }
                        }
                    }
                    SherpaModelKind::Qwen3Asr => {
                        // NOTE: Qwen3-ASR auto-detects language; no per-call hint
                        // slot in OfflineQwen3ASRModelConfig — language field does
                        // not exist on this struct.

                        // Same LLM-decoder context-budget concern as FunASR-Nano
                        // (same Qwen3 family). Cap: 48 entries × 500 total chars.
                        // 48 was chosen as a safer interim over FunASR-Nano's 32 —
                        // it adds headroom for phonetic aliases without hitting EOS
                        // truncation on typical 30s audio. Run the bench harness
                        // (HANDY_QWEN3_HOTWORDS_BENCH=1) to find the true threshold.
                        // TODO(bench): empirically probe N=16,32,48,64,96,128 hotwords
                        //   at 30 s audio → find threshold where output truncates.
                        const QWEN3_HOTWORDS_MAX_ENTRIES: usize = 48;
                        const QWEN3_HOTWORDS_MAX_CHARS: usize = 500;

                        let settings = get_settings(&self.app_handle);

                        let hotwords_str: Option<String> = if settings.custom_words.is_empty() {
                            None
                        } else {
                            let mut acc = String::new();
                            let mut count = 0usize;
                            for w in &settings.custom_words {
                                let w = w.trim();
                                if w.is_empty() {
                                    continue;
                                }
                                if count >= QWEN3_HOTWORDS_MAX_ENTRIES {
                                    break;
                                }
                                let projected_len =
                                    acc.chars().count() + w.chars().count() + 1; // +1 for \n
                                if projected_len > QWEN3_HOTWORDS_MAX_CHARS {
                                    break;
                                }
                                if !acc.is_empty() {
                                    acc.push('\n');
                                }
                                acc.push_str(w);
                                count += 1;
                            }
                            if acc.is_empty() {
                                None
                            } else {
                                debug!(
                                    "Qwen3-ASR: hotwords {} entries / {} chars (capped from {} total)",
                                    count,
                                    acc.chars().count(),
                                    settings.custom_words.len()
                                );
                                Some(acc)
                            }
                        };

                        // File layout verified against upstream Python example
                        // (python-api-examples/offline-qwen3-asr-decode-files.py):
                        //   conv_frontend.onnx  — no int8 suffix on conv_frontend
                        //   encoder.int8.onnx
                        //   decoder.int8.onnx
                        //   tokenizer/          — a subdirectory (not Qwen3-0.6B)
                        config.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
                            conv_frontend: Some(
                                model_path
                                    .join("conv_frontend.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            encoder: Some(
                                model_path
                                    .join("encoder.int8.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            decoder: Some(
                                model_path
                                    .join("decoder.int8.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            tokenizer: Some(
                                model_path
                                    .join("tokenizer")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            // CRITICAL: override landmine defaults (128 / 512) that
                            // silently truncate long audio (same gotcha as FunASR-Nano).
                            max_new_tokens: 4096,
                            max_total_len: 8192,
                            // temperature: keep at 1e-6 (upstream default from
                            // python-api-examples/offline-qwen3-asr-decode-files.py).
                            // Setting 0.0 risks sampler degeneracy. 1e-6 is effectively
                            // greedy while remaining numerically stable.
                            hotwords: hotwords_str,
                            ..Default::default()
                        };
                    }
                    SherpaModelKind::FunAsrNano => {
                        // Bisect step 2 (2026-05-03): re-introduce hotwords
                        // with hard caps. Bisect confirmed step 1 (language +
                        // max_new_tokens) is safe; the long-audio empty-output
                        // regression came from injecting all 78 custom_words
                        // into the Qwen3-0.6B prompt — combined with a long
                        // audio embedding it exhausts the LLM context budget
                        // and the model emits an immediate-EOS token.
                        // Cap: 32 entries × 500 total chars keeps the prompt
                        // small enough to coexist with ~30s of audio.
                        const FUNASR_HOTWORDS_MAX_ENTRIES: usize = 32;
                        const FUNASR_HOTWORDS_MAX_CHARS: usize = 500;

                        let settings = get_settings(&self.app_handle);
                        let lang_hint: Option<String> = match settings.selected_language.as_str() {
                            "auto" => None,
                            "zh" | "zh-Hans" | "zh-Hant" => Some("zh".into()),
                            "en" => Some("en".into()),
                            "ja" => Some("ja".into()),
                            "ko" => Some("ko".into()),
                            "yue" => Some("yue".into()),
                            other => Some(other.to_string()),
                        };

                        // Hotwords: pick the first N non-empty entries that
                        // fit within the char budget. Order = user's list
                        // order — they put the most important words first.
                        let hotwords_str: Option<String> = if settings.custom_words.is_empty() {
                            None
                        } else {
                            let mut acc = String::new();
                            let mut count = 0usize;
                            for w in &settings.custom_words {
                                let w = w.trim();
                                if w.is_empty() {
                                    continue;
                                }
                                if count >= FUNASR_HOTWORDS_MAX_ENTRIES {
                                    break;
                                }
                                let projected_len =
                                    acc.chars().count() + w.chars().count() + 1; // +1 for \n
                                if projected_len > FUNASR_HOTWORDS_MAX_CHARS {
                                    break;
                                }
                                if !acc.is_empty() {
                                    acc.push('\n');
                                }
                                acc.push_str(w);
                                count += 1;
                            }
                            if acc.is_empty() {
                                None
                            } else {
                                debug!(
                                    "FunASR-Nano: hotwords {} entries / {} chars (capped from {} total)",
                                    count,
                                    acc.chars().count(),
                                    settings.custom_words.len()
                                );
                                Some(acc)
                            }
                        };

                        config.model_config.funasr_nano = OfflineFunASRNanoModelConfig {
                            encoder_adaptor: Some(
                                model_path
                                    .join("encoder_adaptor.int8.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            llm: Some(
                                model_path
                                    .join("llm.int8.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            embedding: Some(
                                model_path
                                    .join("embedding.int8.onnx")
                                    .to_string_lossy()
                                    .into_owned(),
                            ),
                            tokenizer: Some(
                                model_path.join("Qwen3-0.6B").to_string_lossy().into_owned(),
                            ),
                            // Cap LLM output to prevent runaway repetition loops.
                            // 512 tokens covers ~120 Chinese chars / ~50 English
                            // words — enough for ~30s of speech.
                            max_new_tokens: 512,
                            language: lang_hint,
                            hotwords: hotwords_str,
                            ..Default::default()
                        };
                    }
                }

                let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
                    let error_msg =
                        format!("Failed to create sherpa-onnx recognizer for {}", model_id);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;

                LoadedEngine::Sherpa(SherpaSession { recognizer, kind })
            }
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded.
    ///
    /// Default strategy: `apple-speech` is the macOS default (streaming partials,
    /// zero-download, ideal for IME/dictation). FunASR-Nano is a user-chosen
    /// "accuracy-first" alternative that must be explicitly selected in Models settings.
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading || self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            let settings = get_settings(&self_clone.app_handle);
            let model_to_load = settings.selected_model.clone();

            if let Err(e) = self_clone.load_model(&model_to_load) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    /// Transcribe audio, optionally overriding the language from settings.
    /// Pass `override_language = None` to use the language stored in settings.
    pub fn transcribe_with_language_override(
        &self,
        audio: Vec<f32>,
        override_language: Option<String>,
    ) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();

        debug!("Audio vector length: {}", audio.len());

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        let settings = get_settings(&self.app_handle);
        let effective = resolve_effective_settings(&settings);

        // Use override language if provided, otherwise fall back to settings.
        let language_to_use =
            override_language.unwrap_or_else(|| settings.selected_language.clone());

        // Validate selected language against the model's supported languages.
        let validated_language = if language_to_use == "auto" {
            "auto".to_string()
        } else {
            let is_supported = self
                .model_manager
                .get_model_info(&settings.selected_model)
                .map(|info| {
                    info.supported_languages.is_empty()
                        || info.supported_languages.contains(&language_to_use)
                })
                .unwrap_or(true);

            if is_supported {
                language_to_use.clone()
            } else {
                warn!(
                    "Language '{}' not supported by current model, falling back to auto-detect",
                    language_to_use
                );
                "auto".to_string()
            }
        };

        let partial_emit_handle = self.app_handle.clone();

        let result = {
            let mut engine_guard = self.lock_engine();
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };
            drop(engine_guard);

            let transcribe_result = catch_unwind(AssertUnwindSafe(
                || -> Result<transcribe_rs::TranscriptionResult> {
                    self.do_transcribe(
                        &mut engine,
                        &audio,
                        &validated_language,
                        &settings,
                        partial_emit_handle.clone(),
                    )
                },
            ));

            match transcribe_result {
                Ok(inner_result) => {
                    let mut engine_guard = self.lock_engine();
                    *engine_guard = Some(engine);
                    inner_result?
                }
                Err(panic_payload) => {
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }
                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );
                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            }
        };

        let engine_ms = st.elapsed().as_millis();

        let skip_word_correction = self
            .model_manager
            .get_model_info(&settings.selected_model)
            .map(|info| {
                matches!(
                    info.engine_type,
                    EngineType::Whisper | EngineType::AppleSpeech
                )
            })
            .unwrap_or(false);

        let t_custom_words = std::time::Instant::now();
        let corrected_result = if !settings.custom_words.is_empty() && !skip_word_correction {
            apply_custom_words(
                &result.text,
                &settings.custom_words,
                settings.word_correction_threshold,
            )
        } else {
            result.text
        };
        let custom_words_ms = t_custom_words.elapsed().as_millis();

        let t_aliases = std::time::Instant::now();
        let aliased_result = if !settings.custom_word_aliases.is_empty() {
            crate::audio_toolkit::apply_word_aliases(&corrected_result, &settings.custom_word_aliases)
        } else {
            corrected_result
        };
        let aliases_ms = t_aliases.elapsed().as_millis();

        let t_filter = std::time::Instant::now();
        let filtered_result = filter_transcription_output(
            &aliased_result,
            &settings.app_language,
            &settings.custom_filler_words,
        );
        let filter_ms = t_filter.elapsed().as_millis();

        // Apply CT-Transformer Chinese punctuation when enabled and language is Chinese.
        // Use effective.punc_zh_enabled so profile overrides are honoured.
        let t_punc = std::time::Instant::now();
        let final_result = apply_punc_zh_if_applicable(
            filtered_result,
            &validated_language,
            &settings.app_language,
            effective.punc_zh_enabled,
            &self.app_handle,
        );
        let punc_ms = t_punc.elapsed().as_millis();

        let et = std::time::Instant::now();
        let total_ms = (et - st).as_millis();
        info!(
            "Transcription (with language override) completed in {}ms",
            total_ms
        );
        debug!(
            "Pipeline timing (override): engine={}ms custom_words={}ms aliases={}ms filter={}ms punc={}ms total={}ms",
            engine_ms, custom_words_ms, aliases_ms, filter_ms, punc_ms, total_ms
        );

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!("Transcription result: {}", final_result);
        }

        self.maybe_unload_immediately("transcription");
        Ok(final_result)
    }

    /// Internal helper: run the engine-specific transcription logic.
    /// Called by both `transcribe` and `transcribe_with_language_override`.
    #[allow(clippy::too_many_arguments)]
    fn do_transcribe(
        &self,
        engine: &mut LoadedEngine,
        audio: &[f32],
        validated_language: &str,
        settings: &crate::settings::AppSettings,
        partial_emit_handle: AppHandle,
    ) -> Result<transcribe_rs::TranscriptionResult> {
        match engine {
            LoadedEngine::Whisper(whisper_engine) => {
                let whisper_language = if validated_language == "auto" {
                    None
                } else {
                    let normalized =
                        if validated_language == "zh-Hans" || validated_language == "zh-Hant" {
                            "zh".to_string()
                        } else {
                            validated_language.to_string()
                        };
                    Some(normalized)
                };
                let params = WhisperInferenceParams {
                    language: whisper_language,
                    translate: settings.translate_to_english,
                    initial_prompt: if settings.custom_words.is_empty() {
                        None
                    } else {
                        Some(settings.custom_words.join(", "))
                    },
                    ..Default::default()
                };
                whisper_engine
                    .transcribe_with(audio, &params)
                    .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))
            }
            LoadedEngine::Parakeet(parakeet_engine) => {
                let params = ParakeetParams {
                    timestamp_granularity: Some(TimestampGranularity::Segment),
                    ..Default::default()
                };
                parakeet_engine
                    .transcribe_with(audio, &params)
                    .map_err(|e| anyhow::anyhow!("Parakeet transcription failed: {}", e))
            }
            LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                .transcribe(audio, &TranscribeOptions::default())
                .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
            LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                .transcribe(audio, &TranscribeOptions::default())
                .map_err(|e| anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)),
            LoadedEngine::SenseVoice(sense_voice_engine) => {
                let language = match validated_language {
                    "zh" | "zh-Hans" | "zh-Hant" => Some("zh".to_string()),
                    "en" => Some("en".to_string()),
                    "ja" => Some("ja".to_string()),
                    "ko" => Some("ko".to_string()),
                    "yue" => Some("yue".to_string()),
                    _ => None,
                };
                let params = SenseVoiceParams {
                    language,
                    // ITN is handled downstream by itn_zh.rs;
                    // disable here to prevent double-processing.
                    use_itn: Some(false),
                };
                sense_voice_engine
                    .transcribe_with(audio, &params)
                    .map_err(|e| anyhow::anyhow!("SenseVoice transcription failed: {}", e))
                    .map(|mut r| {
                        // Strip SenseVoice meta/emotion/language tags
                        // (e.g. <|HAPPY|>, <|zh|>, <|EMO_SAD|>) before
                        // any further processing.
                        r.text = crate::audio_toolkit::sense_voice_filter::strip_meta(&r.text);
                        r
                    })
            }
            LoadedEngine::GigaAM(gigaam_engine) => gigaam_engine
                .transcribe(audio, &TranscribeOptions::default())
                .map_err(|e| anyhow::anyhow!("GigaAM transcription failed: {}", e)),
            LoadedEngine::Canary(canary_engine) => {
                let lang = if validated_language == "auto" {
                    None
                } else {
                    Some(validated_language.to_string())
                };
                let options = TranscribeOptions {
                    language: lang,
                    translate: settings.translate_to_english,
                    ..Default::default()
                };
                canary_engine
                    .transcribe(audio, &options)
                    .map_err(|e| anyhow::anyhow!("Canary transcription failed: {}", e))
            }
            LoadedEngine::Cohere(cohere_engine) => {
                let lang = if validated_language == "auto" {
                    None
                } else if validated_language == "zh-Hans" || validated_language == "zh-Hant" {
                    Some("zh".to_string())
                } else {
                    Some(validated_language.to_string())
                };
                let options = TranscribeOptions {
                    language: lang,
                    ..Default::default()
                };
                cohere_engine
                    .transcribe(audio, &options)
                    .map_err(|e| anyhow::anyhow!("Cohere transcription failed: {}", e))
            }
            #[cfg(target_os = "macos")]
            LoadedEngine::AppleSpeech { .. } => {
                let bcp47 = map_to_bcp47(validated_language);
                let contextual: Vec<String> = settings.custom_words.clone();
                let require_on_device = settings.apple_speech_require_on_device;

                info!(
                    "Apple Speech: locale={} require_on_device={}",
                    bcp47, require_on_device
                );

                let partial_app_handle = partial_emit_handle.clone();
                let bcp47_clone = bcp47.clone();
                let contextual_clone = contextual.clone();
                let first_result = crate::apple_speech::transcribe_with_partials(
                    audio,
                    16000.0,
                    &bcp47_clone,
                    &contextual_clone,
                    require_on_device,
                    30_000,
                    move |text| {
                        debug!("Apple Speech partial: {}", text);
                        let _ = partial_app_handle
                            .emit("transcription-partial", serde_json::json!({ "text": text }));
                    },
                );

                // Helper: returns true for errors that should NOT trigger a network fallback.
                let is_fatal_error = |msg: &str| {
                    msg.contains("Apple Speech permission not granted")
                        || msg.contains("Apple Speech authorization timed out")
                };

                match first_result {
                    Ok(text) => Ok(transcribe_rs::TranscriptionResult {
                        text,
                        segments: None,
                    }),
                    Err(ref e) if require_on_device && !is_fatal_error(e) => {
                        warn!(
                            "Apple Speech on-device attempt failed ({}); retrying with network recognition",
                            e
                        );
                        let partial_app_handle2 = partial_emit_handle.clone();
                        crate::apple_speech::transcribe_with_partials(
                            audio,
                            16000.0,
                            &bcp47,
                            &contextual,
                            false,
                            30_000,
                            move |text| {
                                debug!("Apple Speech partial (network): {}", text);
                                let _ = partial_app_handle2.emit(
                                    "transcription-partial",
                                    serde_json::json!({ "text": text }),
                                );
                            },
                        )
                        .map(|text| transcribe_rs::TranscriptionResult {
                            text,
                            segments: None,
                        })
                        .map_err(|e2| {
                            error!("Apple Speech network fallback also failed: {}", e2);
                            anyhow::anyhow!(
                                "Apple Speech transcription failed (on-device and network both unavailable): {}",
                                e2
                            )
                        })
                    }
                    Err(e) => {
                        error!("Apple Speech transcription failed: {}", e);
                        Err(anyhow::anyhow!("Apple Speech transcription failed: {}", e))
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            LoadedEngine::AppleSpeech { .. } => {
                Err(anyhow::anyhow!("Apple Speech is only available on macOS"))
            }
            LoadedEngine::Sherpa(session) => {
                // VERIFIED Qwen3 → custom_words → punc_zh pipeline:
                // skip_word_correction only gates Whisper | AppleSpeech; Sherpa
                // (including Qwen3Asr) flows through apply_custom_words + punc_zh
                // in the post-processing pipeline above do_transcribe.

                // For Qwen3-ASR: use VAD-aware chunking on long audio.
                // Other Sherpa models (SenseVoice, FunASR-Nano) use the direct path.
                if matches!(session.kind, SherpaModelKind::Qwen3Asr)
                    && audio.len() > 16_000 * 90
                {
                    // Long audio path: split at VAD boundaries, ≤45 s per chunk.
                    let t0 = std::time::Instant::now();
                    let vad_path_result = self
                        .app_handle
                        .path()
                        .resolve(
                            "resources/models/silero_vad_v4.onnx",
                            tauri::path::BaseDirectory::Resource,
                        );

                    // Constants used across all chunking paths
                    const FRAME_SAMPLES: usize = 480; // 30 ms @ 16 kHz
                    const MAX_CHUNK_SAMPLES: usize = 16_000 * 45; // 45 s

                    let chunks: Vec<Vec<f32>> = match vad_path_result {
                        Ok(vad_path) => {
                            match crate::audio_toolkit::SileroVad::new(&vad_path, 0.3) {
                                Ok(mut vad) => {
                                    // Segment the full audio into speech chunks using
                                    // the same 30 ms frame size Silero was trained on.
                                    let mut all_chunks: Vec<Vec<f32>> = Vec::new();
                                    let mut current_chunk: Vec<f32> = Vec::new();

                                    let frames = audio.chunks(FRAME_SAMPLES);
                                    for frame in frames {
                                        if frame.len() < FRAME_SAMPLES {
                                            // Trailing partial frame — append to current
                                            current_chunk.extend_from_slice(frame);
                                            continue;
                                        }
                                        let is_speech = vad
                                            .is_voice(frame)
                                            .unwrap_or(true); // on error, treat as speech

                                        if is_speech {
                                            current_chunk.extend_from_slice(frame);
                                            // Flush when chunk hits max size
                                            if current_chunk.len() >= MAX_CHUNK_SAMPLES {
                                                all_chunks.push(std::mem::take(&mut current_chunk));
                                            }
                                        } else {
                                            // Silence boundary: if current chunk is
                                            // substantial (>300 ms), flush it.
                                            if current_chunk.len() >= FRAME_SAMPLES * 10 {
                                                all_chunks.push(std::mem::take(&mut current_chunk));
                                            }
                                            // otherwise accumulate small leftovers
                                        }
                                    }
                                    if !current_chunk.is_empty() {
                                        all_chunks.push(current_chunk);
                                    }

                                    if all_chunks.is_empty() {
                                        debug!("Qwen3-ASR: VAD found no speech in long audio");
                                        return Ok(transcribe_rs::TranscriptionResult {
                                            text: String::new(),
                                            segments: None,
                                        });
                                    }
                                    debug!(
                                        "Qwen3-ASR: split {}s audio into {} VAD chunks",
                                        audio.len() / 16_000,
                                        all_chunks.len()
                                    );
                                    all_chunks
                                }
                                Err(e) => {
                                    warn!(
                                        "Qwen3-ASR: VAD init failed ({}); falling back to naive 45s split",
                                        e
                                    );
                                    audio
                                        .chunks(MAX_CHUNK_SAMPLES)
                                        .map(|c| c.to_vec())
                                        .collect()
                                }
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Qwen3-ASR: VAD model path resolution failed ({}); falling back to naive 45s split",
                                e
                            );
                            audio
                                .chunks(MAX_CHUNK_SAMPLES)
                                .map(|c| c.to_vec())
                                .collect()
                        }
                    };

                    let n_chunks = chunks.len();
                    let mut parts: Vec<String> = Vec::with_capacity(n_chunks);

                    for (i, chunk) in chunks.iter().enumerate() {
                        // Emit progress event so the overlay can show progress
                        let elapsed_ms = t0.elapsed().as_millis() as u64;
                        let _ = partial_emit_handle.emit(
                            "transcription-progress",
                            serde_json::json!({
                                "phase": "qwen3_chunk",
                                "current": i,
                                "total": n_chunks,
                                "elapsed_ms": elapsed_ms,
                            }),
                        );

                        let stream = session.recognizer.create_stream();
                        stream.accept_waveform(16000, chunk.as_slice());
                        session.recognizer.decode(&stream);
                        let chunk_text =
                            stream.get_result().map(|r| r.text).unwrap_or_default();

                        if !chunk_text.is_empty() {
                            parts.push(chunk_text);
                        }
                        debug!(
                            "Qwen3-ASR chunk {}/{}: {} chars in {}ms",
                            i + 1,
                            n_chunks,
                            parts.last().map(|s| s.chars().count()).unwrap_or(0),
                            t0.elapsed().as_millis()
                        );
                    }

                    // Final progress event (done)
                    let _ = partial_emit_handle.emit(
                        "transcription-progress",
                        serde_json::json!({
                            "phase": "qwen3_chunk",
                            "current": n_chunks,
                            "total": n_chunks,
                            "elapsed_ms": t0.elapsed().as_millis() as u64,
                        }),
                    );

                    let text = parts.join(" ");
                    Ok(transcribe_rs::TranscriptionResult {
                        text,
                        segments: None,
                    })
                } else {
                    // Short audio path (or non-Qwen3 Sherpa): direct transcription.
                    let stream = session.recognizer.create_stream();
                    stream.accept_waveform(16000, audio);
                    session.recognizer.decode(&stream);
                    let text = stream.get_result().map(|r| r.text).unwrap_or_default();
                    Ok(transcribe_rs::TranscriptionResult {
                        text,
                        segments: None,
                    })
                }
            }
        }
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();

        debug!("Audio vector length: {}", audio.len());

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);
        let effective = resolve_effective_settings(&settings);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        let validated_language = if settings.selected_language == "auto" {
            "auto".to_string()
        } else {
            let is_supported = self
                .model_manager
                .get_model_info(&settings.selected_model)
                .map(|info| {
                    info.supported_languages.is_empty()
                        || info
                            .supported_languages
                            .contains(&settings.selected_language)
                })
                .unwrap_or(true);

            if is_supported {
                settings.selected_language.clone()
            } else {
                warn!(
                    "Language '{}' not supported by current model, falling back to auto-detect",
                    settings.selected_language
                );
                "auto".to_string()
            }
        };

        // Clone app_handle for use inside the catch_unwind closure (AppleSpeech partial emitter).
        let partial_emit_handle = self.app_handle.clone();

        // Perform transcription with the appropriate engine.
        // We use catch_unwind to prevent engine panics from poisoning the mutex,
        // which would make the app hang indefinitely on subsequent operations.
        let result = {
            let mut engine_guard = self.lock_engine();

            // Take the engine out so we own it during transcription.
            // If the engine panics, we simply don't put it back (effectively unloading it)
            // instead of poisoning the mutex.
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };

            // Release the lock before transcribing — no mutex held during the engine call
            drop(engine_guard);

            let transcribe_result = catch_unwind(AssertUnwindSafe(
                || -> Result<transcribe_rs::TranscriptionResult> {
                    self.do_transcribe(
                        &mut engine,
                        &audio,
                        &validated_language,
                        &settings,
                        partial_emit_handle.clone(),
                    )
                },
            ));

            match transcribe_result {
                Ok(inner_result) => {
                    // Success or normal error — put the engine back
                    let mut engine_guard = self.lock_engine();
                    *engine_guard = Some(engine);
                    inner_result?
                }
                Err(panic_payload) => {
                    // Engine panicked — do NOT put it back (it's in an unknown state).
                    // The engine is dropped here, effectively unloading it.
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );

                    // Clear the model ID so it will be reloaded on next attempt
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }

                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );

                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            }
        };

        let engine_ms = st.elapsed().as_millis();

        // Apply word correction if custom words are configured.
        // Skip for Whisper (custom words passed as initial_prompt) and Apple Speech
        // (custom words passed as contextual hints to SFSpeechRecognizer).
        let skip_word_correction = self
            .model_manager
            .get_model_info(&settings.selected_model)
            .map(|info| {
                matches!(
                    info.engine_type,
                    EngineType::Whisper | EngineType::AppleSpeech
                )
            })
            .unwrap_or(false);

        let t_custom_words = std::time::Instant::now();
        let corrected_result = if !settings.custom_words.is_empty() && !skip_word_correction {
            apply_custom_words(
                &result.text,
                &settings.custom_words,
                settings.word_correction_threshold,
            )
        } else {
            result.text
        };
        let custom_words_ms = t_custom_words.elapsed().as_millis();

        // Apply phonetic alias substitutions (exact substring, longer-first).
        let t_aliases = std::time::Instant::now();
        let aliased_result = if !settings.custom_word_aliases.is_empty() {
            crate::audio_toolkit::apply_word_aliases(&corrected_result, &settings.custom_word_aliases)
        } else {
            corrected_result
        };
        let aliases_ms = t_aliases.elapsed().as_millis();

        // Filter out filler words and hallucinations
        let t_filter = std::time::Instant::now();
        let filtered_result = filter_transcription_output(
            &aliased_result,
            &settings.app_language,
            &settings.custom_filler_words,
        );
        let filter_ms = t_filter.elapsed().as_millis();

        // Apply CT-Transformer Chinese punctuation when enabled and language is Chinese.
        // Use effective.punc_zh_enabled so profile overrides are honoured.
        let t_punc = std::time::Instant::now();
        let final_result = apply_punc_zh_if_applicable(
            filtered_result,
            &validated_language,
            &settings.app_language,
            effective.punc_zh_enabled,
            &self.app_handle,
        );
        let punc_ms = t_punc.elapsed().as_millis();

        let et = std::time::Instant::now();
        let total_ms = (et - st).as_millis();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        info!(
            "Transcription completed in {}ms{}",
            total_ms, translation_note
        );
        debug!(
            "Pipeline timing: engine={}ms custom_words={}ms aliases={}ms filter={}ms punc={}ms total={}ms",
            engine_ms, custom_words_ms, aliases_ms, filter_ms, punc_ms, total_ms
        );

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!("Transcription result: {}", final_result);
        }

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }
}

/// Apply CT-Transformer Chinese punctuation restoration if the conditions are met.
///
/// Returns the punctuated text on success or `text` unchanged on any error/miss.
/// Never panics — all errors are logged and gracefully skipped.
fn apply_punc_zh_if_applicable(
    text: String,
    validated_language: &str,
    app_language: &str,
    punc_zh_enabled: bool,
    app_handle: &tauri::AppHandle,
) -> String {
    if !punc_zh_enabled || text.is_empty() {
        debug!(
            "punc_zh: skipped (enabled={} empty={})",
            punc_zh_enabled,
            text.is_empty()
        );
        return text;
    }

    // Determine whether the *effective* language is Chinese.
    // We check `validated_language` first (explicitly selected by user or model).
    // When it is "auto" we fall back to `app_language`.
    let lang_to_check = if validated_language == "auto" {
        app_language
    } else {
        validated_language
    };

    let base_lang = lang_to_check
        .split(&['-', '_'][..])
        .next()
        .unwrap_or(lang_to_check);

    let language_says_zh = base_lang == "zh" || lang_to_check == "yue";

    // Content-based fallback: when language metadata is ambiguous (e.g. Apple
    // Speech preset with `auto` + non-Chinese app_language), inspect the
    // transcription itself. The CT-Transformer-Punc model we ship is the
    // zh-en vocab272727 variant — it punctuates Chinese text safely and is
    // a no-op on pure ASCII, so applying it whenever any CJK char appears is
    // both safe and correct.
    let text_has_cjk = text.chars().any(|c| {
        let cp = c as u32;
        // CJK Unified Ideographs core + extension A + Compatibility + general
        // CJK punctuation ranges. Covers 簡/繁 + Cantonese + Japanese kanji.
        (0x3400..=0x4DBF).contains(&cp)         // CJK Ext A
            || (0x4E00..=0x9FFF).contains(&cp)  // CJK Unified Ideographs
            || (0xF900..=0xFAFF).contains(&cp)  // CJK Compatibility Ideographs
            || (0x3000..=0x303F).contains(&cp) // CJK Symbols and Punctuation
    });

    debug!(
        "punc_zh: validated_lang={} app_lang={} lang_to_check={} language_says_zh={} text_has_cjk={} text_len={}",
        validated_language,
        app_language,
        lang_to_check,
        language_says_zh,
        text_has_cjk,
        text.len()
    );

    if !language_says_zh && !text_has_cjk {
        // Neither metadata nor content suggests Chinese — skip punc layer
        debug!(
            "punc_zh: skipped — neither language nor content suggests Chinese (text='{}')",
            text.chars().take(50).collect::<String>()
        );
        return text;
    }

    // Density short-circuit: if the upstream engine (e.g. FunASR-Nano LLM) already
    // produced punctuation, running CT-Punc again is wasted work that can also
    // distort spacing around existing marks. Threshold 2% is empirically the
    // floor below which a Chinese sentence almost certainly lacks proper marks.
    let total_chars = text.chars().count();
    if total_chars > 0 {
        let punct_count = text
            .chars()
            .filter(|c| {
                matches!(
                    *c,
                    '。' | '，'
                        | '！'
                        | '？'
                        | '；'
                        | '：'
                        | '、'
                        | '\u{201C}'
                        | '\u{201D}'
                        | '\u{2018}'
                        | '\u{2019}'
                        | '.'
                        | ','
                        | '!'
                        | '?'
                        | ';'
                        | ':'
                )
            })
            .count();
        let density = punct_count as f32 / total_chars as f32;
        if density >= 0.02 {
            debug!(
                "punc_zh: skipped (already punctuated, density={:.3} {}/{})",
                density, punct_count, total_chars
            );
            return text;
        }
    }

    // Build path: <app_data_dir>/models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8
    let model_dir = match crate::portable::app_data_dir(app_handle) {
        Ok(d) => d
            .join("models")
            .join("sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"),
        Err(e) => {
            warn!("punc_zh: cannot resolve app_data_dir: {}", e);
            return text;
        }
    };

    match crate::audio_toolkit::punc_zh::add_punctuation(&model_dir, &text) {
        Ok(punctuated) => {
            debug!("punc_zh: applied punctuation");
            punctuated
        }
        Err(e) => {
            // Model absent or inference failed — silently degrade
            debug!("punc_zh: skipped ({})", e);
            text
        }
    }
}

/// Map an app-internal language code (ISO 639-1 or zh-Hans/zh-Hant) to a BCP-47
/// locale tag suitable for SFSpeechRecognizer.
///
/// SFSpeechRecognizer requires full BCP-47 tags (e.g. "en-US") while the rest of
/// Handy uses short ISO 639-1 codes (e.g. "en"). This function bridges the two.
pub fn map_to_bcp47(lang: &str) -> String {
    if lang == "auto" {
        // Apple Speech does not have a true "auto" locale. Inherit the macOS
        // system language so a Chinese-system Mac actually transcribes Chinese
        // instead of running everything through en-US.
        let system_locale = tauri_plugin_os::locale().unwrap_or_default();
        let lower = system_locale.replace('_', "-").to_lowercase();
        let base = lower.split('-').next().unwrap_or("");
        return match base {
            "zh" => {
                if lower.contains("hant") || lower.contains("tw") || lower.contains("hk") {
                    "zh-TW".to_string()
                } else {
                    "zh-CN".to_string()
                }
            }
            "ja" => "ja-JP".to_string(),
            "ko" => "ko-KR".to_string(),
            "fr" => "fr-FR".to_string(),
            "de" => "de-DE".to_string(),
            "es" => "es-ES".to_string(),
            "pt" => "pt-BR".to_string(),
            "ru" => "ru-RU".to_string(),
            "it" => "it-IT".to_string(),
            _ => "en-US".to_string(),
        };
    }
    match lang {
        "en" => "en-US".to_string(),
        "zh" | "zh-Hans" => "zh-CN".to_string(),
        "zh-Hant" => "zh-TW".to_string(),
        "ja" => "ja-JP".to_string(),
        "ko" => "ko-KR".to_string(),
        "fr" => "fr-FR".to_string(),
        "de" => "de-DE".to_string(),
        "es" => "es-ES".to_string(),
        "pt" => "pt-BR".to_string(),
        "ru" => "ru-RU".to_string(),
        "it" => "it-IT".to_string(),
        "nl" => "nl-NL".to_string(),
        "pl" => "pl-PL".to_string(),
        "tr" => "tr-TR".to_string(),
        "ar" => "ar-SA".to_string(),
        "hi" => "hi-IN".to_string(),
        "th" => "th-TH".to_string(),
        "vi" => "vi-VN".to_string(),
        "id" => "id-ID".to_string(),
        "ms" => "ms-MY".to_string(),
        "uk" => "uk-UA".to_string(),
        "cs" => "cs-CZ".to_string(),
        "sk" => "sk-SK".to_string(),
        "ro" => "ro-RO".to_string(),
        "hu" => "hu-HU".to_string(),
        "fi" => "fi-FI".to_string(),
        "da" => "da-DK".to_string(),
        "sv" => "sv-SE".to_string(),
        "nb" | "no" => "nb-NO".to_string(),
        "el" => "el-GR".to_string(),
        "he" => "he-IL".to_string(),
        "bg" => "bg-BG".to_string(),
        "hr" => "hr-HR".to_string(),
        "ca" => "ca-ES".to_string(),
        // Pass-through if already a full BCP-47 tag (contains a hyphen)
        s if s.contains('-') => s.to_string(),
        // Fallback: append -XX region as best guess
        s => format!("{}-{}", s, s.to_uppercase()),
    }
}

/// Apply the user's accelerator preferences to the transcribe-rs global atomics.
/// Called on startup and whenever the user changes the setting.
pub fn apply_accelerator_settings(app: &tauri::AppHandle) {
    use transcribe_rs::accel;

    let settings = get_settings(app);

    let whisper_pref = match settings.whisper_accelerator {
        WhisperAcceleratorSetting::Auto => accel::WhisperAccelerator::Auto,
        WhisperAcceleratorSetting::Cpu => accel::WhisperAccelerator::CpuOnly,
        WhisperAcceleratorSetting::Gpu => accel::WhisperAccelerator::Gpu,
    };
    accel::set_whisper_accelerator(whisper_pref);
    accel::set_whisper_gpu_device(settings.whisper_gpu_device);
    info!(
        "Whisper accelerator set to: {}, gpu_device: {}",
        whisper_pref,
        if settings.whisper_gpu_device == accel::GPU_DEVICE_AUTO {
            "auto".to_string()
        } else {
            settings.whisper_gpu_device.to_string()
        }
    );

    let ort_pref = match settings.ort_accelerator {
        OrtAcceleratorSetting::Auto => accel::OrtAccelerator::Auto,
        OrtAcceleratorSetting::Cpu => accel::OrtAccelerator::CpuOnly,
        OrtAcceleratorSetting::Cuda => accel::OrtAccelerator::Cuda,
        OrtAcceleratorSetting::DirectMl => accel::OrtAccelerator::DirectMl,
        OrtAcceleratorSetting::Rocm => accel::OrtAccelerator::Rocm,
    };
    accel::set_ort_accelerator(ort_pref);
    info!("ORT accelerator set to: {}", ort_pref);
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: i32,
    pub name: String,
    pub total_vram_mb: usize,
}

static GPU_DEVICES: OnceLock<Vec<GpuDeviceOption>> = OnceLock::new();

fn cached_gpu_devices() -> &'static [GpuDeviceOption] {
    use transcribe_rs::whisper_cpp::gpu::list_gpu_devices;

    GPU_DEVICES.get_or_init(|| {
        // ggml's Vulkan backend uses FMA3 instructions internally.
        // On older CPUs without FMA3 (e.g. Sandy Bridge Xeons) this causes
        // a SIGILL crash that cannot be caught. Skip enumeration entirely
        // on those CPUs — GPU-accelerated whisper won't work there anyway.
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("fma") {
            warn!("CPU lacks FMA3 support — skipping GPU device enumeration");
            return Vec::new();
        }

        list_gpu_devices()
            .into_iter()
            .map(|d| GpuDeviceOption {
                id: d.id,
                name: d.name,
                total_vram_mb: d.total_vram / (1024 * 1024),
            })
            .collect()
    })
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub whisper: Vec<String>,
    pub ort: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return which accelerators are compiled into this build.
pub fn get_available_accelerators() -> AvailableAccelerators {
    use transcribe_rs::accel::OrtAccelerator;

    let ort_options: Vec<String> = OrtAccelerator::available()
        .into_iter()
        .map(|a| a.to_string())
        .collect();

    let whisper_options = vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()];

    AvailableAccelerators {
        whisper: whisper_options,
        ort: ort_options,
        gpu_devices: cached_gpu_devices().to_vec(),
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so engine's strong_count is always >= 2 while the
        // watcher is alive. When it reaches 1, only this instance remains
        // and we can safely shut down.
        if Arc::strong_count(&self.engine) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully
        if let Some(handle) = self.watcher_handle.lock().unwrap().take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen3-ASR hotwords cap bench harness
//
// Run: HANDY_QWEN3_HOTWORDS_BENCH=1 cargo test qwen3_hotwords_bench -- --nocapture
//
// The test loads the Qwen3-ASR model from $HANDY_QWEN3_MODEL_DIR and a 30 s
// reference .wav file from $HANDY_QWEN3_BENCH_WAV, then probes hotwords counts
// N = 16, 32, 48, 64, 96, 128 to find the threshold where output truncates.
//
// EOS truncation is suspected when:
//   output_chars < expected_chars * 0.5
// where expected_chars is from the N=16 baseline run.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod qwen3_bench {
    use super::*;

    /// Hotwords bench — skipped unless HANDY_QWEN3_HOTWORDS_BENCH=1
    #[test]
    #[ignore] // run with: cargo test qwen3_hotwords_bench -- --ignored --nocapture
    fn qwen3_hotwords_bench() {
        if std::env::var("HANDY_QWEN3_HOTWORDS_BENCH").as_deref() != Ok("1") {
            eprintln!("Skipped — set HANDY_QWEN3_HOTWORDS_BENCH=1 to run");
            return;
        }

        let model_dir = std::env::var("HANDY_QWEN3_MODEL_DIR")
            .expect("HANDY_QWEN3_MODEL_DIR must point to the extracted model directory");
        let wav_path = std::env::var("HANDY_QWEN3_BENCH_WAV")
            .expect("HANDY_QWEN3_BENCH_WAV must point to a 16 kHz mono WAV file (>=30 s)");

        let audio = crate::audio_toolkit::read_wav_samples(&wav_path)
            .expect("Failed to read bench WAV");

        // Generate a synthetic vocabulary of N unique single-character Chinese words
        let full_vocab: Vec<String> = (0x4E00_u32..0x4E00 + 200)
            .map(|cp| char::from_u32(cp).unwrap().to_string())
            .collect();

        let probe_counts: [u32; 6] = [16, 32, 48, 64, 96, 128];
        let mut baseline_chars: Option<usize> = None;

        for &n in &probe_counts {
            let hotwords_str = {
                let words: Vec<&str> = full_vocab
                    .iter()
                    .take(n as usize)
                    .map(|s| s.as_str())
                    .collect();
                Some(words.join("\n"))
            };

            let model_path = std::path::Path::new(&model_dir);
            let config = sherpa_onnx::OfflineRecognizerConfig {
                model_config: sherpa_onnx::OfflineModelConfig {
                    qwen3_asr: sherpa_onnx::OfflineQwen3ASRModelConfig {
                        conv_frontend: Some(
                            model_path.join("conv_frontend.onnx").to_string_lossy().into_owned(),
                        ),
                        encoder: Some(
                            model_path.join("encoder.int8.onnx").to_string_lossy().into_owned(),
                        ),
                        decoder: Some(
                            model_path.join("decoder.int8.onnx").to_string_lossy().into_owned(),
                        ),
                        tokenizer: Some(
                            model_path.join("tokenizer").to_string_lossy().into_owned(),
                        ),
                        max_new_tokens: 4096,
                        max_total_len: 8192,
                        hotwords: hotwords_str,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            };

            let recognizer = match sherpa_onnx::OfflineRecognizer::new(&config) {
                Some(r) => r,
                None => {
                    eprintln!("N={n}: OfflineRecognizer::new returned None — THRESHOLD FOUND");
                    break;
                }
            };

            let stream = recognizer.create_stream();
            stream.accept_waveform(16000, &audio);
            recognizer.decode(&stream);
            let text = stream.get_result().map(|r| r.text).unwrap_or_default();
            let char_count = text.chars().count();

            if baseline_chars.is_none() {
                baseline_chars = Some(char_count);
            }
            let expected = baseline_chars.unwrap_or(1);
            let truncated = char_count < expected / 2;

            eprintln!(
                "N={n:4}: {char_count:5} chars  truncated={}  text={:.80}",
                truncated,
                &text[..text.len().min(80)],
            );

            if truncated {
                eprintln!(">>> EOS truncation suspected at N={n}. Recommended cap = {}", n / 2);
                break;
            }
        }
    }
}
