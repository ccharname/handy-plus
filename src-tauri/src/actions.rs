#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use crate::observability::{self, Outcome, RequestId, Stage, Stopwatch};
use crate::output;
use crate::profile_resolver::{resolve_effective_settings, EffectiveSettings};
use crate::settings::{get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tauri::Manager;
use tauri::{AppHandle, Emitter};

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

/// Run a single LLM post-processing step with an explicit prompt string.
/// Returns `Some(text)` on success, `None` on hard failure (no output / provider error).
async fn run_single_llm_step(
    settings: &AppSettings,
    effective_provider_id: &str,
    prompt: &str,
    text_input: &str,
) -> Option<String> {
    let provider = match settings
        .post_process_provider(effective_provider_id)
        .cloned()
    {
        Some(provider) => provider,
        None => match settings.active_post_process_provider().cloned() {
            Some(p) => p,
            None => {
                debug!("Post-processing enabled but no provider is selected");
                return None;
            }
        },
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Disable reasoning for providers where post-processing rarely benefits from it.
    // - custom: top-level reasoning_effort (works for local OpenAI-compat servers)
    // - openrouter: nested reasoning object; exclude:true also keeps reasoning text
    //   out of the response so it can't pollute structured-output JSON parsing
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(prompt);
        let user_content = text_input.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            reasoning_effort.clone(),
            reasoning.clone(),
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return Some(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            return Some(strip_invisible_chars(&content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return Some(strip_invisible_chars(&content));
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let processed_prompt = prompt.replace("${output}", text_input);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        reasoning_effort,
        reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&content);
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            Some(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

pub(crate) async fn post_process_transcription(
    settings: &AppSettings,
    transcription: &str,
    effective: Option<&EffectiveSettings>,
) -> Option<String> {
    // Use effective overrides if available (Power Mode), otherwise use global settings.
    let effective_provider_id = effective
        .map(|e| e.post_process_provider_id.as_str())
        .unwrap_or(&settings.post_process_provider_id);

    // Determine if we should use a chain or single-prompt mode.
    // Chain mode: effective (profile override) or settings.post_process_chain is Some(vec) with at least one entry.
    // Some(empty vec) in profile means "disable chain for this profile".
    // Single-prompt mode: everything else (backward-compat).
    let chain = effective
        .and_then(|e| e.post_process_chain.as_deref())
        .or(settings.post_process_chain.as_deref())
        .filter(|v| !v.is_empty());

    if let Some(prompt_ids) = chain {
        debug!("Post-process chain mode: {} step(s)", prompt_ids.len());
        let mut current_text = transcription.to_string();

        for (step_idx, prompt_id) in prompt_ids.iter().enumerate() {
            // Look up the prompt template.
            let prompt_template = match settings
                .post_process_prompts
                .iter()
                .find(|p| &p.id == prompt_id)
            {
                Some(p) => p.prompt.clone(),
                None => {
                    warn!(
                        "Chain step {}: prompt_id '{}' not found — skipping",
                        step_idx + 1,
                        prompt_id
                    );
                    continue;
                }
            };

            match run_single_llm_step(
                settings,
                effective_provider_id,
                &prompt_template,
                &current_text,
            )
            .await
            {
                Some(result) => {
                    debug!(
                        "Chain step {} ('{}') succeeded. Output length: {} chars",
                        step_idx + 1,
                        prompt_id,
                        result.len()
                    );
                    current_text = result;
                }
                None => {
                    warn!(
                        "Chain step {} ('{}') failed — skipping, keeping previous text",
                        step_idx + 1,
                        prompt_id
                    );
                    // skip this step: current_text unchanged
                }
            }
        }

        // Return Some only if the text actually changed (i.e. at least one step succeeded).
        if current_text != transcription {
            Some(current_text)
        } else {
            None
        }
    } else {
        // ── Single-prompt mode (backward-compat) ──────────────────────────────
        // Resolve which prompt to use: profile override > global setting
        let effective_prompt_id = effective
            .and_then(|e| e.post_process_selected_prompt_id.as_deref())
            .or(settings.post_process_selected_prompt_id.as_deref());

        let selected_prompt_id = match effective_prompt_id {
            Some(id) => id.to_string(),
            None => {
                debug!("Post-processing skipped because no prompt is selected");
                return None;
            }
        };

        let prompt = match settings
            .post_process_prompts
            .iter()
            .find(|prompt| prompt.id == selected_prompt_id)
        {
            Some(prompt) => prompt.prompt.clone(),
            None => {
                debug!(
                    "Post-processing skipped because prompt '{}' was not found",
                    selected_prompt_id
                );
                return None;
            }
        };

        run_single_llm_step(settings, effective_provider_id, &prompt, transcription).await
    }
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
}

/// Try to archive `text` as a diary entry.
///
/// Returns `true` if a keyword was matched (regardless of whether the write
/// succeeded — the caller always pastes the full text either way).
pub(crate) fn maybe_archive_diary(settings: &crate::settings::AppSettings, text: &str) -> bool {
    let diary_dir = match settings.diary_dir.as_deref() {
        Some(d) if !d.is_empty() => d,
        _ => return false,
    };

    // Trim whitespace before matching.
    let trimmed = text.trim();

    // Find which keyword (if any) is a prefix of the text, followed by optional
    // punctuation (space, comma, period, CJK full-stop, CJK comma, colon, etc.).
    let matched_keyword = settings.diary_keywords.iter().find_map(|kw| {
        let lower_text = trimmed.to_lowercase();
        let lower_kw = kw.to_lowercase();
        if lower_text.starts_with(&lower_kw) {
            let rest = &trimmed[lower_kw.len()..];
            // Allow zero or more punctuation/separator chars between keyword and body.
            let sep_len = rest
                .chars()
                .take_while(|c| {
                    matches!(
                        *c,
                        ' ' | '\t'
                            | ','
                            | '，'
                            | '。'
                            | '.'
                            | '、'
                            | '：'
                            | ':'
                            | '!'
                            | '！'
                            | '？'
                            | '?'
                    )
                })
                .map(|c| c.len_utf8())
                .sum::<usize>();
            Some((kw.len(), lower_kw.len() + sep_len))
        } else {
            None
        }
    });

    let body_offset = match matched_keyword {
        Some((_, offset)) => offset,
        None => return false,
    };

    let body = trimmed[body_offset..].trim();

    // Expand leading ~ in the path.
    let expanded_dir = if diary_dir.starts_with('~') {
        match dirs_next::home_dir() {
            Some(home) => home
                .join(&diary_dir[2..]) // skip "~/"
                .to_string_lossy()
                .to_string(),
            None => diary_dir.to_string(),
        }
    } else {
        diary_dir.to_string()
    };

    let dir_path = std::path::Path::new(&expanded_dir);

    if let Err(e) = std::fs::create_dir_all(dir_path) {
        warn!("diary: failed to create directory {}: {}", expanded_dir, e);
        return true; // keyword matched even though write failed
    }

    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let file_path = dir_path.join(format!("{}.md", today));

    let time_str = chrono::Local::now().format("%H:%M").to_string();
    let entry = format!("## {}\n{}\n\n", time_str, body);

    use std::io::Write;
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
    {
        Ok(mut f) => {
            if let Err(e) = f.write_all(entry.as_bytes()) {
                warn!("diary: failed to write entry to {:?}: {}", file_path, e);
            } else {
                debug!("diary: appended entry to {:?}", file_path);
            }
        }
        Err(e) => {
            warn!("diary: failed to open {:?}: {}", file_path, e);
        }
    }

    true
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    effective: &EffectiveSettings,
) -> ProcessedTranscription {
    let settings = get_settings(app);

    // Use effective language for Chinese variant conversion
    let mut effective_settings_for_variant = settings.clone();
    effective_settings_for_variant.selected_language = effective.selected_language.clone();

    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;

    if let Some(converted_text) =
        maybe_convert_chinese_variant(&effective_settings_for_variant, transcription).await
    {
        final_text = converted_text;
    }

    if post_process {
        if let Some(processed_text) =
            post_process_transcription(&settings, &final_text, Some(effective)).await
        {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            // Determine which prompt was used (profile override > global setting)
            let prompt_id = effective
                .post_process_selected_prompt_id
                .as_deref()
                .or(settings.post_process_selected_prompt_id.as_deref());

            if let Some(pid) = prompt_id {
                if let Some(prompt) = settings.post_process_prompts.iter().find(|p| p.id == pid) {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    // Diary archival: runs on the raw transcription (before post-processing) so
    // that the keyword is still present. The paste pipeline is not affected.
    maybe_archive_diary(&settings, transcription);

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // t0_hotkey: from keydown dispatch into this handler.
        // Mint a fresh request id and persist it so stop() can reference it.
        let req = RequestId::new();
        if let Some(active) = app.try_state::<crate::observability::ActiveRequestId>() {
            active.set(req);
        }
        let t0_sw = Stopwatch::start();

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        show_recording_overlay(app);

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        // ── FD-006 M2: enable chunked streaming for qwen3_mlx preset ─────────
        // If the active preset is qwen3_mlx and chunked streaming is enabled,
        // wire the AudioChunker + StreamingOrchestrator + drainer *before*
        // try_start_recording so the first audio frame is caught.
        let chunked_streaming_enabled = settings.qwen3_mlx_streaming_chunked
            && settings
                .active_preset_id
                .as_deref()
                .map(|id| id == "qwen3_mlx")
                .unwrap_or(false);

        if chunked_streaming_enabled {
            let tm_for_cb = app.state::<Arc<TranscriptionManager>>();
            let tm_clone = Arc::clone(&tm_for_cb);
            rm.start_chunked_streaming(move |delta: &str| {
                tm_clone.append_incremental_paste(delta);
            });
            debug!("[FD-006 M2] Chunked streaming enabled for qwen3_mlx");
        }
        // ── End FD-006 M2 wiring ───────────────────────────────────────────

        let mut recording_error: Option<String> = None;
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
            }
        }

        if recording_error.is_none() {
            // t0_hotkey: dispatch succeeded, recording has started
            observability::ok_with(
                req,
                Stage::T0Hotkey,
                t0_sw.elapsed_ms(),
                serde_json::json!({
                    "binding_id": binding_id
                }),
            );
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            // t0_hotkey: dispatch failed (microphone error etc.)
            observability::record_stage(
                req,
                Stage::T0Hotkey,
                Outcome::Error,
                t0_sw.elapsed_ms(),
                Some(
                    serde_json::json!({ "binding_id": binding_id, "error": "recording_start_failed" }),
                ),
            );
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            utils::hide_recording_overlay(app);
            change_tray_icon(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        // Capture a per-request-id from the app state if available;
        // fall back to a fresh one so stop() is always instrumented.
        let req = app
            .try_state::<crate::observability::ActiveRequestId>()
            .map(|s| s.get())
            .unwrap_or_else(RequestId::new);
        let total_sw = Stopwatch::start();

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );
            let req_async = req;
            let total_sw_async = total_sw;

            // Resolve Power Mode effective settings as early as possible, while the
            // foreground app is still the one the user was dictating into.
            let effective = resolve_effective_settings(&get_settings(&ah));
            debug!(
                "Power Mode resolved: profile={:?} lang={} paste={:?} model={}",
                effective.matched_profile_name,
                effective.selected_language,
                effective.paste_method,
                effective.selected_model
            );

            // Hot-swap the loaded engine if the resolved profile asked for a
            // different model. Costs 1-3s while the new model loads, so this
            // path is gated by `profile_hot_swap_engine` (off by default) and
            // only fires when the global model differs from the resolved one.
            {
                let current = tm.get_current_model();
                if current.as_deref() != Some(effective.selected_model.as_str()) {
                    let target = effective.selected_model.clone();
                    let tm_for_load = Arc::clone(&tm);
                    match tauri::async_runtime::spawn_blocking(move || {
                        // FD-003 M3.5 #8: P-core binding for model-swap worker.
                        crate::platform::elevate_thread_qos("model-hotswap");
                        tm_for_load.load_model(&target)
                    })
                    .await
                    {
                        Ok(Ok(())) => debug!(
                            "Hot-swapped engine to '{}' for profile {:?}",
                            effective.selected_model, effective.matched_profile_name
                        ),
                        Ok(Err(e)) => warn!(
                            "Profile model swap to '{}' failed: {}; keeping current engine",
                            effective.selected_model, e
                        ),
                        Err(e) => warn!("Profile model swap task panicked: {}", e),
                    }
                }
            }

            // Clear any partial text from a previous session as soon as we enter
            // the transcribing state.
            let _ = ah.emit("transcription-partial-clear", ());
            // Also reset the incremental-paste cursor so the new run starts
            // from an empty baseline (Apple Speech path only; no-op for others).
            tm.reset_incremental_paste();

            let stop_recording_time = Instant::now();
            let t2_sw = Stopwatch::start();
            // FD-006 M2: signal the streaming session to drain + finalize.
            // stop_chunked_streaming(true) waits for the drainer thread to
            // process remaining partials before we proceed to transcription.
            if rm.is_streaming_active() {
                rm.stop_chunked_streaming(true);
                debug!("[FD-006 M2] Chunked streaming session drained");
            }
            if let Some(samples) = rm.stop_recording(&binding_id) {
                let recording_ms = t2_sw.elapsed_ms();
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    // t2_recording: zero samples → cancelled
                    observability::record_stage(
                        req_async,
                        Stage::T2Recording,
                        Outcome::Cancelled,
                        recording_ms,
                        Some(serde_json::json!({ "sample_count": 0 })),
                    );
                    observability::cancelled(req_async, Stage::Total, total_sw_async.elapsed_ms());
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // t2_recording: samples acquired
                    // Sample rate is assumed 16kHz (after resample); compute duration_ms.
                    let sample_count = samples.len();
                    let duration_ms_recording = (sample_count as f64 / 16_000.0) * 1000.0;
                    observability::ok_with(
                        req_async,
                        Stage::T2Recording,
                        recording_ms,
                        serde_json::json!({
                            "sample_count": sample_count,
                            "audio_duration_ms": duration_ms_recording as u64
                        }),
                    );

                    // t3_vad + t4_resample are measured inside the recording consumer
                    // thread and cannot be extracted without significant refactor.
                    // Emit placeholder records with audio duration so handy-logs
                    // can compute derived ratios (vad_ms/30s, resample_ms/30s).
                    // Real per-frame timing improvement is tracked in M4.
                    observability::ok_with(
                        req_async,
                        Stage::T3Vad,
                        0.0,
                        serde_json::json!({
                            "audio_duration_ms": duration_ms_recording as u64,
                            "note": "inline_with_recording"
                        }),
                    );
                    observability::ok_with(
                        req_async,
                        Stage::T4Resample,
                        0.0,
                        serde_json::json!({
                            "audio_duration_ms": duration_ms_recording as u64,
                            "note": "inline_with_recording"
                        }),
                    );

                    // Save WAV concurrently with transcription
                    let file_name = format!("v2t-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        // FD-003 M3.5 #8: P-core binding for WAV-save worker.
                        crate::platform::elevate_thread_qos("wav-save");
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // t5_inference: model inference
                    let t5_sw = Stopwatch::start();
                    // Transcribe concurrently with WAV save
                    let transcription_time = Instant::now();
                    let transcription_result = tm.transcribe(samples);

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    match transcription_result {
                        Ok(transcription) => {
                            let t5_ms = t5_sw.elapsed_ms();
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription
                            );

                            // t5_inference: success
                            {
                                let char_count = transcription.chars().count();
                                // RTF: inference_ms / audio_duration_ms
                                let rtf = if duration_ms_recording > 0.0 {
                                    t5_ms / duration_ms_recording
                                } else {
                                    0.0
                                };
                                let mut t5_extra = serde_json::json!({
                                    "inference_ms": t5_ms as u64,
                                    "audio_duration_ms": duration_ms_recording as u64,
                                    "rtf": format!("{:.3}", rtf),
                                    "transcript_char_count": char_count
                                });
                                if observability::log_transcripts() {
                                    if let Some(obj) = t5_extra.as_object_mut() {
                                        obj.insert(
                                            "transcript".to_string(),
                                            serde_json::Value::String(transcription.clone()),
                                        );
                                    }
                                }
                                observability::ok_with(
                                    req_async,
                                    Stage::T5Inference,
                                    t5_ms,
                                    t5_extra,
                                );
                            }

                            // Transcription is done; clear any partial text from the overlay.
                            let _ = ah.emit("transcription-partial-clear", ());

                            if post_process {
                                show_processing_overlay(&ah);
                            }
                            // t6_postprocess
                            let t6_sw = Stopwatch::start();
                            let processed = process_transcription_output(
                                &ah,
                                &transcription,
                                post_process,
                                &effective,
                            )
                            .await;
                            let t6_ms = t6_sw.elapsed_ms();
                            {
                                let final_char_count = processed.final_text.chars().count();
                                let post_processed = processed.post_processed_text.is_some();
                                observability::ok_with(
                                    req_async,
                                    Stage::T6Postprocess,
                                    t6_ms,
                                    serde_json::json!({
                                        "post_processed": post_processed,
                                        "final_char_count": final_char_count
                                    }),
                                );
                            }

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                // Extract effective paste overrides for the closure.
                                let eff_paste_method = effective
                                    .matched_profile_name
                                    .as_ref()
                                    .map(|_| effective.paste_method);
                                let eff_trailing_space = effective
                                    .matched_profile_name
                                    .as_ref()
                                    .map(|_| effective.append_trailing_space);
                                let eff_auto_submit = effective
                                    .matched_profile_name
                                    .as_ref()
                                    .map(|_| effective.auto_submit);

                                // For Apple Speech incremental paste: take the cursor
                                // (cumulative text already pasted during recognition) and
                                // compute the residual — only that suffix still needs to
                                // be passed to paste_with_overrides so that trailing-space
                                // and auto-submit are applied correctly.
                                // For all other engines the cursor is empty, so the full
                                // final_text is pasted as before.
                                let already_pasted = tm.take_incremental_paste_cursor();
                                let text_to_paste = if already_pasted.is_empty() {
                                    // Normal path: no incremental paste happened.
                                    final_text.clone()
                                } else if final_text.starts_with(&already_pasted) {
                                    // Clean case: paste only the residual suffix.
                                    let residual = final_text[already_pasted.len()..].to_string();
                                    debug!(
                                        "Incremental paste finalisation: {} chars already pasted, {} residual",
                                        already_pasted.len(),
                                        residual.len()
                                    );
                                    residual
                                } else {
                                    // The final text diverged from what was already pasted
                                    // (e.g. post-processing rewrote it, or partial stream
                                    // had a retroactive edit we couldn't follow). Erase the
                                    // already-pasted prefix via backspaces, then paste the
                                    // full final. Without erasing, the user sees a duplicate.
                                    let backspace_count = already_pasted.chars().count();
                                    warn!(
                                        "Final text diverged from incremental cursor; \
                                         erasing {} chars via backspace then pasting full final \
                                         (already_pasted={:?}, final_text={:?})",
                                        backspace_count,
                                        already_pasted.chars().take(20).collect::<String>(),
                                        final_text.chars().take(20).collect::<String>()
                                    );
                                    if let Some(enigo_state) =
                                        ah_clone.try_state::<crate::input::EnigoState>()
                                    {
                                        if let Ok(mut enigo) = enigo_state.0.lock() {
                                            if let Err(e) = crate::input::send_backspaces(
                                                &mut enigo,
                                                backspace_count,
                                            ) {
                                                error!(
                                                    "Failed to erase already-pasted prefix: {}",
                                                    e
                                                );
                                            }
                                        }
                                    }
                                    final_text.clone()
                                };

                                let req_for_paste = req_async;
                                let total_sw_for_paste = total_sw_async;
                                ah.run_on_main_thread(move || {
                                    // M2.5: Route output through the streaming sink.
                                    //
                                    // For the current C2 batch path (both SenseVoice and
                                    // qwen3_mlx) `text_to_paste` arrives as a complete
                                    // string.  We call `sink.finalize()` which performs a
                                    // one-shot paste — same end result as `paste_with_overrides`
                                    // but routed through the appropriate mechanism
                                    // (Accessibility / Keystroke / Clipboard) based on the
                                    // frontmost app.
                                    //
                                    // When C3 streaming (token-by-token qwen3_mlx) is wired,
                                    // `do_transcribe` will call `sink.append(delta)` per token
                                    // and `text_to_paste` arriving here will be empty (or just
                                    // the post-processed suffix), handled by the same finalize().
                                    let t7_sw = Stopwatch::start();

                                    // Append trailing space if configured (mirrors
                                    // paste_with_overrides behaviour).
                                    let text_with_space = match eff_trailing_space {
                                        Some(true) => format!("{} ", text_to_paste),
                                        _ => {
                                            let settings = crate::settings::get_settings(&ah_clone);
                                            if settings.append_trailing_space {
                                                format!("{} ", text_to_paste)
                                            } else {
                                                text_to_paste.clone()
                                            }
                                        }
                                    };

                                    // When Power Mode has a paste_method override that is
                                    // None (suppress paste), skip the sink entirely.
                                    use crate::settings::PasteMethod;
                                    let suppress_paste = eff_paste_method == Some(PasteMethod::None);

                                    // Compute sink kind upfront so observability log below sees
                                    // the actual runtime sink kind, not a constant placeholder.
                                    let (sink_kind_for_obs, paste_outcome): (&'static str, Result<(), String>) = if suppress_paste {
                                        debug!("[T7] PasteMethod::None — suppressing output sink");
                                        ("suppressed", Ok(()))
                                    } else if text_with_space.is_empty() {
                                        // Nothing to paste (e.g. incremental path already
                                        // delivered all text); still fire auto-submit below.
                                        ("empty", Ok(()))
                                    } else {
                                        // Select the best sink for the frontmost app.
                                        let mut sink = output::select_sink_auto();
                                        let sink_kind = sink.kind_str();

                                        // For the batch path, append the full text then finalize.
                                        let outcome = sink.append(&text_with_space)
                                            .and_then(|()| sink.finalize())
                                            .map_err(|e| {
                                                // If the preferred sink failed, fall back to
                                                // paste_with_overrides (legacy clipboard path).
                                                warn!(
                                                    "[T7] sink {:?} failed ({}); falling back to clipboard paste",
                                                    sink_kind, e
                                                );
                                                e
                                            })
                                            // On sink failure, fall through to legacy path.
                                            .or_else(|_| {
                                                crate::clipboard::paste_with_overrides(
                                                    text_with_space.clone(),
                                                    ah_clone.clone(),
                                                    eff_paste_method,
                                                    // trailing space already applied above
                                                    Some(false),
                                                    Some(false),
                                                )
                                            });
                                        (sink_kind, outcome)
                                    };

                                    let t7_ms = t7_sw.elapsed_ms();

                                    // Fire auto-submit if configured (after paste).
                                    // Use paste_with_overrides with empty text + auto_submit=true
                                    // so that the Return key fires without re-pasting.
                                    if paste_outcome.is_ok() {
                                        let should_auto_submit = match eff_auto_submit {
                                            Some(v) => v,
                                            None => {
                                                let settings =
                                                    crate::settings::get_settings(&ah_clone);
                                                settings.auto_submit
                                            }
                                        };
                                        let auto_submit_method = eff_paste_method.unwrap_or_else(
                                            || crate::settings::get_settings(&ah_clone).paste_method,
                                        );
                                        if should_auto_submit
                                            && auto_submit_method != PasteMethod::None
                                        {
                                            // Send Return key via enigo directly.
                                            if let Some(enigo_state) =
                                                ah_clone.try_state::<crate::input::EnigoState>()
                                            {
                                                if let Ok(mut enigo) = enigo_state.0.lock() {
                                                    use enigo::{Direction, Keyboard, Key};
                                                    let settings = crate::settings::get_settings(
                                                        &ah_clone,
                                                    );
                                                    match settings.auto_submit_key {
                                                        crate::settings::AutoSubmitKey::Enter => {
                                                            let _ = enigo.key(
                                                                Key::Return,
                                                                Direction::Click,
                                                            );
                                                        }
                                                        crate::settings::AutoSubmitKey::CtrlEnter => {
                                                            let _ = enigo.key(
                                                                Key::Control,
                                                                Direction::Press,
                                                            );
                                                            let _ = enigo.key(
                                                                Key::Return,
                                                                Direction::Click,
                                                            );
                                                            let _ = enigo.key(
                                                                Key::Control,
                                                                Direction::Release,
                                                            );
                                                        }
                                                        crate::settings::AutoSubmitKey::CmdEnter => {
                                                            let _ = enigo
                                                                .key(Key::Meta, Direction::Press);
                                                            let _ = enigo.key(
                                                                Key::Return,
                                                                Direction::Click,
                                                            );
                                                            let _ = enigo.key(
                                                                Key::Meta,
                                                                Direction::Release,
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    match paste_outcome {
                                        Ok(()) => {
                                            debug!(
                                                "Text pasted successfully in {:?}",
                                                paste_time.elapsed()
                                            );
                                            observability::ok_with(
                                                req_for_paste,
                                                Stage::T7Output,
                                                t7_ms,
                                                serde_json::json!({
                                                    "sink_kind": sink_kind_for_obs,
                                                    "paste_lag_ms": t7_ms as u64
                                                }),
                                            );
                                            observability::ok_with(req_for_paste, Stage::Total,
                                                total_sw_for_paste.elapsed_ms(),
                                                serde_json::json!({ "end_to_end_ms": total_sw_for_paste.elapsed_ms() as u64 }));
                                        }
                                        Err(e) => {
                                            error!("Failed to paste transcription: {}", e);
                                            observability::error(req_for_paste, Stage::T7Output, t7_ms);
                                            observability::record_stage(req_for_paste, Stage::Total,
                                                Outcome::Error, total_sw_for_paste.elapsed_ms(), None);
                                            let _ = ah_clone.emit("paste-error", ());
                                        }
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            let t5_ms = t5_sw.elapsed_ms();
                            debug!("Global Shortcut Transcription error: {}", err);
                            // t5_inference: error path
                            observability::record_stage(
                                req_async,
                                Stage::T5Inference,
                                Outcome::Error,
                                t5_ms,
                                Some(serde_json::json!({ "error": err.to_string() })),
                            );
                            observability::record_stage(
                                req_async,
                                Stage::Total,
                                Outcome::Error,
                                total_sw_async.elapsed_ms(),
                                None,
                            );
                            // Clear any partial text that may have accumulated before the error.
                            let _ = ah.emit("transcription-partial-clear", ());
                            // Also clear the incremental-paste cursor so the next run
                            // does not inherit stale state from a failed session.
                            tm.reset_incremental_paste();
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                // t2_recording: recording was cancelled or produced nothing
                observability::cancelled(
                    req_async,
                    Stage::T2Recording,
                    total_sw_async.elapsed_ms(),
                );
                observability::cancelled(req_async, Stage::Total, total_sw_async.elapsed_ms());
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        // Handy+: post-processing is always-on when a prompt is selected
        // (post_process_selected_prompt_id != None).  Power Mode profiles for
        // code-style apps opt out by setting post_process_selected_prompt_id = None.
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

// ─────────────────────────────────────────────────────────────────────────────
// Diary archival — unit tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod diary_tests {
    use super::*;
    use crate::settings::{get_default_settings, AppSettings};

    /// Build a minimal settings object with diary fields set.
    fn build_settings(diary_dir: Option<&str>, keywords: Vec<&str>) -> AppSettings {
        let mut s = get_default_settings();
        s.diary_dir = diary_dir.map(|d| d.to_string());
        s.diary_keywords = keywords.into_iter().map(|k| k.to_string()).collect();
        s
    }

    // ── 1. Basic Chinese keyword trigger ─────────────────────────────────────
    #[test]
    fn test_basic_chinese_diary_trigger() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        let settings = build_settings(Some(path), vec!["日记"]);

        let matched = maybe_archive_diary(&settings, "日记 今天阳光真好");
        assert!(matched, "Chinese keyword '日记' should trigger archival");

        let entries: Vec<_> = std::fs::read_dir(path)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1, "exactly one diary file should be created");
    }

    // ── 2. Multiple CJK/ASCII separator characters ────────────────────────────
    #[test]
    fn test_multiple_separators() {
        // All separator chars explicitly listed in maybe_archive_diary
        let separators: &[&str] = &[
            " ", "\t", ",", "，", "。", ".", "、", "：", ":", "!", "！", "？", "?",
        ];
        for sep in separators {
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().to_str().unwrap();
            let settings = build_settings(Some(path), vec!["备忘"]);
            let text = format!("备忘{}测试内容", sep);
            assert!(
                maybe_archive_diary(&settings, &text),
                "separator {:?} should trigger archival",
                sep
            );
        }
    }

    // ── 3. Case-insensitive English keyword ───────────────────────────────────
    #[test]
    fn test_case_insensitive_english() {
        for prefix in ["diary", "Diary", "DIARY", "dIaRy"] {
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().to_str().unwrap();
            let settings = build_settings(Some(path), vec!["diary"]);
            let text = format!("{} testing today", prefix);
            assert!(
                maybe_archive_diary(&settings, &text),
                "prefix {:?} should match case-insensitively",
                prefix
            );
        }
    }

    // ── 4. No keyword match → returns false, no file ──────────────────────────
    #[test]
    fn test_no_match_returns_false() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = build_settings(Some(dir.path().to_str().unwrap()), vec!["日记"]);

        let matched = maybe_archive_diary(&settings, "今天天气真好");
        assert!(!matched, "unrelated text should not match");

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 0, "no file should be created on non-match");
    }

    // ── 5. diary_dir = None → disabled, returns false ─────────────────────────
    #[test]
    fn test_diary_dir_none_skips() {
        let settings = build_settings(None, vec!["日记"]);
        assert!(
            !maybe_archive_diary(&settings, "日记 测试"),
            "should return false when diary_dir is None"
        );
    }

    // ── 6. diary_dir = empty string → disabled ────────────────────────────────
    #[test]
    fn test_diary_dir_empty_string_skips() {
        let settings = build_settings(Some(""), vec!["日记"]);
        assert!(
            !maybe_archive_diary(&settings, "日记 测试"),
            "should return false when diary_dir is empty string"
        );
    }

    // ── 7. Tilde (~) expansion ────────────────────────────────────────────────
    #[test]
    fn test_tilde_expansion() {
        let subdir_name = format!("handy_test_diary_{}", std::process::id());
        let tilde_path = format!("~/{}", subdir_name);
        let settings = build_settings(Some(&tilde_path), vec!["日记"]);

        let matched = maybe_archive_diary(&settings, "日记 测试展开");
        assert!(matched, "tilde path should be accepted and keyword matched");

        // Cleanup
        if let Some(home) = dirs_next::home_dir() {
            let expanded = home.join(&subdir_name);
            std::fs::remove_dir_all(&expanded).ok();
        }
    }

    // ── 8. File format: ## HH:MM header + body ───────────────────────────────
    #[test]
    fn test_file_format_hh_mm() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = build_settings(Some(dir.path().to_str().unwrap()), vec!["日记"]);

        maybe_archive_diary(&settings, "日记 测试格式");

        let entry = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .expect("diary file should exist");
        let content = std::fs::read_to_string(entry.path()).unwrap();

        assert!(
            content.starts_with("## "),
            "entry should start with markdown h2 header: {:?}",
            content
        );
        // The first line is "## HH:MM" — verify it contains ':'
        let first_line = content.lines().next().unwrap_or("");
        assert!(
            first_line.contains(':'),
            "header should contain HH:MM time with colon: {:?}",
            first_line
        );
        assert!(
            content.contains("测试格式"),
            "body text should be present in file"
        );
    }

    // ── 9. Append mode: two entries → same file ───────────────────────────────
    #[test]
    fn test_append_mode() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = build_settings(Some(dir.path().to_str().unwrap()), vec!["日记"]);

        maybe_archive_diary(&settings, "日记 第一条");
        maybe_archive_diary(&settings, "日记 第二条");

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "both entries should append to same daily file"
        );

        let content = std::fs::read_to_string(entries[0].path()).unwrap();
        assert!(
            content.contains("第一条"),
            "first entry body should be in file"
        );
        assert!(
            content.contains("第二条"),
            "second entry body should be in file"
        );
    }

    // ── 10. Body text stripped of keyword prefix ──────────────────────────────
    #[test]
    fn test_body_excludes_keyword() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = build_settings(Some(dir.path().to_str().unwrap()), vec!["note"]);

        maybe_archive_diary(&settings, "note: important meeting at 3pm");

        let entry = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .expect("diary file should exist");
        let content = std::fs::read_to_string(entry.path()).unwrap();

        assert!(
            content.contains("important meeting at 3pm"),
            "body should contain text after the keyword"
        );
        // The keyword itself should NOT appear in the body (it was the prefix)
        let body_section = content
            .lines()
            .skip(1) // skip ## HH:MM header
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !body_section.trim_start().starts_with("note"),
            "body should not start with the keyword: {:?}",
            body_section
        );
    }

    // ── 11. Keyword at end only (no body) → archived with empty body ──────────
    #[test]
    fn test_keyword_only_no_body() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = build_settings(Some(dir.path().to_str().unwrap()), vec!["日记"]);

        // Just the keyword, no body text
        let matched = maybe_archive_diary(&settings, "日记");
        assert!(matched, "lone keyword should still match");
        // File should be created
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "file should be created even with empty body"
        );
    }

    // ── 12. Multiple keywords configured — first match wins ───────────────────
    #[test]
    fn test_multiple_keywords_configured() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = build_settings(
            Some(dir.path().to_str().unwrap()),
            vec!["日记", "memo", "note"],
        );

        assert!(maybe_archive_diary(&settings, "memo 购物清单"));
        assert!(maybe_archive_diary(&settings, "note buy milk"));
        assert!(maybe_archive_diary(&settings, "日记 第三条"));

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        // All three should append to the same daily file
        assert_eq!(entries.len(), 1, "all entries should go to same daily file");
        let content = std::fs::read_to_string(entries[0].path()).unwrap();
        assert!(content.contains("购物清单"));
        assert!(content.contains("buy milk"));
        assert!(content.contains("第三条"));
    }
}
