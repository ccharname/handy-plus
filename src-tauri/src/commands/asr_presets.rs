use crate::managers::transcription::TranscriptionManager;
use crate::settings::{default_asr_presets, get_settings, write_settings, AsrPreset};
use std::sync::Arc;
use tauri::{AppHandle, State};

/// Single source of truth for which ASR presets a user actually sees.
/// Filters `default_asr_presets()` by:
///   1. macOS 26+ → drop `apple_native` (SFSpeechRecognizer routes through
///      SpeechAnalyzer there and hangs the process — see
///      `is_apple_speech_available` in swift/apple_speech.swift and the
///      matching gotcha memory).
///   2. `experimental_enabled == false` → drop non-builtin presets. Triage
///      2026-05-04 confirmed both experimental presets ship with user-visible
///      regressions today (voxtral: zh-yue → Devanagari, p50 ~10s; qwen3:
///      2.5 GB on disk). Power users opt in via Advanced Settings.
///
/// Both the settings UI (`list_asr_presets`) and the tray submenu must use
/// this helper so a hidden preset can never be invoked from any entry point.
pub fn filtered_asr_presets(app: &AppHandle) -> Vec<AsrPreset> {
    let mut presets = default_asr_presets();
    #[cfg(target_os = "macos")]
    if crate::utils::is_macos_26_or_later() {
        presets.retain(|p| p.id != "apple_native");
    }
    // Voxtral is deprecated end-to-end: Qwen3-ASR-MLX is 6.6× smaller, 8× faster,
    // and actually identifies Cantonese/Shanghainese/Hokkien (Voxtral hallucinates
    // Devanagari or returns empty). The preset definition stays in default_asr_presets
    // so bench / CLI A-B lookups by id still work, but the UI card is gone.
    // Original Intel/Linux/Win gate is now redundant — hidden everywhere.
    presets.retain(|p| p.id != "experimental_voxtral");
    let settings = get_settings(app);
    if !settings.experimental_enabled {
        presets.retain(|p| p.builtin);
    }
    presets
}

#[tauri::command]
#[specta::specta]
pub async fn list_asr_presets(app: AppHandle) -> Result<Vec<AsrPreset>, String> {
    Ok(filtered_asr_presets(&app))
}

#[tauri::command]
#[specta::specta]
pub async fn apply_asr_preset(
    app: AppHandle,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
) -> Result<(), String> {
    let preset = default_asr_presets()
        .into_iter()
        .find(|p| p.id == preset_id)
        .ok_or_else(|| format!("Preset {} not found", preset_id))?;

    // Step 1: write all non-model fields + mark active preset
    let mut settings = get_settings(&app);
    settings.selected_language = preset.language.clone();
    settings.punc_zh_enabled = preset.punc_zh_enabled;
    if let Some(chain) = preset.require_post_process_chain.clone() {
        settings.post_process_chain = Some(chain);
    }
    if let Some(on_device) = preset.require_apple_speech_on_device {
        settings.apple_speech_require_on_device = on_device;
    }
    settings.active_preset_id = Some(preset.id.clone());

    // Step 2: set the model id and persist everything in one write
    settings.selected_model = preset.model_id.clone();
    write_settings(&app, settings);

    // Step 3: load the model (handles unload + load + events)
    let model_id = preset.model_id.clone();
    let tm = Arc::clone(&*transcription_manager);
    tauri::async_runtime::spawn_blocking(move || tm.load_model(&model_id))
        .await
        .map_err(|e| format!("Model load task panicked: {}", e))?
        .map_err(|e| format!("Failed to load preset model: {}", e))?;

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn detach_asr_preset(app: AppHandle) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.active_preset_id = None;
    write_settings(&app, settings);
    Ok(())
}
