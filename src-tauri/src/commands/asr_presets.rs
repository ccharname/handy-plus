use crate::managers::transcription::TranscriptionManager;
use crate::settings::{default_asr_presets, get_settings, write_settings, AsrPreset};
use std::sync::Arc;
use tauri::{AppHandle, State};

/// Single source of truth for which ASR presets a user actually sees.
/// Filters `default_asr_presets()` by platform:
///   - `qwen3_mlx` requires macOS aarch64 (mlx-audio-swift); hidden elsewhere.
///
/// Both the settings UI (`list_asr_presets`) and the tray submenu must use
/// this helper so a platform-gated preset can never be invoked from any entry
/// point on an incompatible platform.
pub fn filtered_asr_presets(_app: &AppHandle) -> Vec<AsrPreset> {
    let mut presets = default_asr_presets();
    // qwen3_mlx is Apple Silicon only — hide on Intel macOS / Linux / Windows.
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        presets.retain(|p| p.id != "qwen3_mlx");
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
