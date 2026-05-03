use crate::audio_toolkit::{apply_custom_words, filter_transcription_output};
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
    OfflineFunASRNanoModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
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
    /// Retained for future per-call language override and diagnostic logging.
    #[allow(dead_code)]
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
    /// Supports SenseVoice and FunASR-Nano model families.
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
                    SherpaModelKind::FunAsrNano => {
                        // CAUTION: keep this block minimal until the long-audio
                        // empty-output regression is bisected. 2026-05-03 logs
                        // showed audio ≥10s returning empty text in ~210ms (RTF
                        // 0.014, well below physical floor) when ANY of
                        // {temperature=0, max_new_tokens=512, hotwords=large
                        // string, language hint} were set. Hypothesis: Qwen3
                        // LLM context budget is exhausted by audio embedding +
                        // hotwords prompt. Restore tuning fields one at a time.
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

        let t_filter = std::time::Instant::now();
        let filtered_result = filter_transcription_output(
            &corrected_result,
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
            "Pipeline timing (override): engine={}ms custom_words={}ms filter={}ms punc={}ms total={}ms",
            engine_ms, custom_words_ms, filter_ms, punc_ms, total_ms
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
                let sherpa_lang = match validated_language {
                    "zh" | "zh-Hans" | "zh-Hant" => "zh",
                    "en" => "en",
                    "ja" => "ja",
                    "ko" => "ko",
                    "yue" => "yue",
                    _ => "auto",
                };
                let stream = session.recognizer.create_stream();
                stream.accept_waveform(16000, audio);
                session.recognizer.decode(&stream);
                let text = stream.get_result().map(|r| r.text).unwrap_or_default();
                let _ = sherpa_lang;
                Ok(transcribe_rs::TranscriptionResult {
                    text,
                    segments: None,
                })
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

        // Filter out filler words and hallucinations
        let t_filter = std::time::Instant::now();
        let filtered_result = filter_transcription_output(
            &corrected_result,
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
            "Pipeline timing: engine={}ms custom_words={}ms filter={}ms punc={}ms total={}ms",
            engine_ms, custom_words_ms, filter_ms, punc_ms, total_ms
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
