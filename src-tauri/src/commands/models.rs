use crate::managers::model::{ModelInfo, ModelManager};
use crate::managers::transcription::{ModelStateEvent, TranscriptionManager};
use crate::settings::{get_settings, write_settings, ModelUnloadTimeout};
use futures_util::StreamExt;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};

#[tauri::command]
#[specta::specta]
pub async fn get_available_models(
    model_manager: State<'_, Arc<ModelManager>>,
) -> Result<Vec<ModelInfo>, String> {
    Ok(model_manager.get_available_models())
}

#[tauri::command]
#[specta::specta]
pub async fn get_model_info(
    model_manager: State<'_, Arc<ModelManager>>,
    model_id: String,
) -> Result<Option<ModelInfo>, String> {
    Ok(model_manager.get_model_info(&model_id))
}

#[tauri::command]
#[specta::specta]
pub async fn download_model(
    app_handle: AppHandle,
    model_manager: State<'_, Arc<ModelManager>>,
    model_id: String,
) -> Result<(), String> {
    let result = model_manager
        .download_model(&model_id)
        .await
        .map_err(|e| e.to_string());

    if let Err(ref error) = result {
        let _ = app_handle.emit(
            "model-download-failed",
            serde_json::json!({ "model_id": &model_id, "error": error }),
        );
    }

    result
}

#[tauri::command]
#[specta::specta]
pub async fn delete_model(
    app_handle: AppHandle,
    model_manager: State<'_, Arc<ModelManager>>,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    model_id: String,
) -> Result<(), String> {
    // If deleting the active model, unload it and clear the setting
    let settings = get_settings(&app_handle);
    if settings.selected_model == model_id {
        transcription_manager
            .unload_model()
            .map_err(|e| format!("Failed to unload model: {}", e))?;

        let mut settings = get_settings(&app_handle);
        settings.selected_model = String::new();
        write_settings(&app_handle, settings);
    }

    model_manager
        .delete_model(&model_id)
        .map_err(|e| e.to_string())
}

/// Shared logic for switching the active model, used by both the Tauri command
/// and the tray menu handler.
///
/// Validates the model, updates the persisted setting, and loads the model
/// unless the unload timeout is set to "Immediately" (in which case the model
/// will be loaded on-demand during the next transcription).
pub fn switch_active_model(app: &AppHandle, model_id: &str) -> Result<(), String> {
    let model_manager = app.state::<Arc<ModelManager>>();
    let transcription_manager = app.state::<Arc<TranscriptionManager>>();

    // Atomically claim the loading slot — prevents concurrent model loads
    // from tray double-clicks or overlapping commands. The guard resets the
    // flag on drop (including early returns, errors, and panics).
    let _loading_guard = transcription_manager
        .try_start_loading()
        .ok_or_else(|| "Model load already in progress".to_string())?;

    // Check if model exists and is available
    let model_info = model_manager
        .get_model_info(model_id)
        .ok_or_else(|| format!("Model not found: {}", model_id))?;

    if !model_info.is_downloaded {
        return Err(format!("Model not downloaded: {}", model_id));
    }

    let settings = get_settings(app);
    let unload_timeout = settings.model_unload_timeout;
    let old_model = settings.selected_model.clone();

    // Persist the new selection early so the frontend sees the correct model
    // when it reacts to events emitted by load_model.
    let mut settings = settings;
    settings.selected_model = model_id.to_string();

    // Reset language to auto if the new model doesn't support the currently selected language.
    // This prevents stale language settings from causing errors (e.g. Canary receiving zh-Hans)
    // and stops downstream processing (e.g. OpenCC) from running on an irrelevant language.
    if settings.selected_language != "auto"
        && !model_info.supported_languages.is_empty()
        && !model_info
            .supported_languages
            .contains(&settings.selected_language)
    {
        log::info!(
            "Resetting language from '{}' to 'auto' (not supported by {})",
            settings.selected_language,
            model_id
        );
        settings.selected_language = "auto".to_string();
    }

    write_settings(app, settings);

    // Skip eager loading if unload is set to "Immediately" — the model
    // will be loaded on-demand during the next transcription.
    if unload_timeout == ModelUnloadTimeout::Immediately {
        // Notify frontend — load_model won't be called so no events
        // would otherwise be emitted.
        let _ = app.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "selection_changed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );
        log::info!(
            "Model selection changed to {} (not loading — unload set to Immediately).",
            model_id
        );
        return Ok(());
    }

    // Load the model. On failure, revert the persisted selection.
    if let Err(e) = transcription_manager.load_model(model_id) {
        let mut settings = get_settings(app);
        settings.selected_model = old_model;
        write_settings(app, settings);
        return Err(e.to_string());
    }

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn set_active_model(
    app_handle: AppHandle,
    _model_manager: State<'_, Arc<ModelManager>>,
    _transcription_manager: State<'_, Arc<TranscriptionManager>>,
    model_id: String,
) -> Result<(), String> {
    switch_active_model(&app_handle, &model_id)
}

#[tauri::command]
#[specta::specta]
pub async fn get_current_model(app_handle: AppHandle) -> Result<String, String> {
    let settings = get_settings(&app_handle);
    Ok(settings.selected_model)
}

#[tauri::command]
#[specta::specta]
pub async fn get_transcription_model_status(
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
) -> Result<Option<String>, String> {
    Ok(transcription_manager.get_current_model())
}

#[tauri::command]
#[specta::specta]
pub async fn is_model_loading(
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
) -> Result<bool, String> {
    // Check if transcription manager has a loaded model
    let current_model = transcription_manager.get_current_model();
    Ok(current_model.is_none())
}

#[tauri::command]
#[specta::specta]
pub async fn has_any_models_available(
    model_manager: State<'_, Arc<ModelManager>>,
) -> Result<bool, String> {
    let models = model_manager.get_available_models();
    Ok(models.iter().any(|m| m.is_downloaded))
}

#[tauri::command]
#[specta::specta]
pub async fn has_any_models_or_downloads(
    model_manager: State<'_, Arc<ModelManager>>,
) -> Result<bool, String> {
    let models = model_manager.get_available_models();
    // Return true if any models are downloaded OR if any downloads are in progress
    Ok(models.iter().any(|m| m.is_downloaded))
}

#[tauri::command]
#[specta::specta]
pub async fn cancel_download(
    model_manager: State<'_, Arc<ModelManager>>,
    model_id: String,
) -> Result<(), String> {
    model_manager
        .cancel_download(&model_id)
        .map_err(|e| e.to_string())
}

// ── CT-Transformer Chinese punctuation model ──────────────────────────────

/// Directory name for the punctuation model (matches the extracted archive name).
const PUNC_MODEL_DIR: &str = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8";

/// Download URL for the punctuation model archive (zh+en, int8 ≈ 62 MB).
const PUNC_MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2";

/// Returns the path to the extracted punctuation model directory.
fn punc_model_dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    crate::portable::app_data_dir(app)
        .map(|d| d.join("models").join(PUNC_MODEL_DIR))
        .map_err(|e| e.to_string())
}

/// Returns `true` when the punctuation model directory exists and contains `model.onnx`.
#[tauri::command]
#[specta::specta]
pub async fn is_punc_downloaded(app: AppHandle) -> bool {
    match punc_model_dir(&app) {
        Ok(dir) => crate::audio_toolkit::punc_zh::is_punc_model_present(&dir),
        Err(_) => false,
    }
}

/// Download and extract the CT-Transformer punctuation model archive.
///
/// Emits `punc-download-progress` events with `{ downloaded, total, percentage }`.
/// Emits `punc-download-failed` on error.
#[tauri::command]
#[specta::specta]
pub async fn download_punc_model(app: AppHandle) -> Result<(), String> {
    use bzip2::read::BzDecoder;
    use std::fs;
    use std::io::Write;
    use std::time::{Duration, Instant};
    use tar::Archive;

    let models_dir = crate::portable::app_data_dir(&app)
        .map(|d| d.join("models"))
        .map_err(|e| e.to_string())?;

    fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;

    let dest_dir = models_dir.join(PUNC_MODEL_DIR);
    if crate::audio_toolkit::punc_zh::is_punc_model_present(&dest_dir) {
        return Ok(());
    }

    let partial_path = models_dir.join(format!("{}.tar.bz2.partial", PUNC_MODEL_DIR));

    let client = reqwest::Client::new();
    let resume_from = if partial_path.exists() {
        partial_path.metadata().map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mut request = client.get(PUNC_MODEL_URL);
    if resume_from > 0 {
        request = request.header("Range", format!("bytes={}-", resume_from));
    }

    let response = request.send().await.map_err(|e| e.to_string())?;

    if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT
    {
        let err = format!("HTTP {}", response.status());
        let _ = app.emit("punc-download-failed", serde_json::json!({ "error": &err }));
        return Err(err);
    }

    let total_size = resume_from + response.content_length().unwrap_or(0);
    let mut downloaded = resume_from;
    let mut stream = response.bytes_stream();

    let mut file = if resume_from > 0 {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&partial_path)
            .map_err(|e| e.to_string())?
    } else {
        std::fs::File::create(&partial_path).map_err(|e| e.to_string())?
    };

    let _ = app.emit(
        "punc-download-progress",
        serde_json::json!({ "downloaded": downloaded, "total": total_size, "percentage": 0.0 }),
    );

    let mut last_emit = Instant::now();
    let throttle = Duration::from_millis(100);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        file.write_all(&chunk).map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;

        if last_emit.elapsed() >= throttle {
            let pct = if total_size > 0 {
                (downloaded as f64 / total_size as f64) * 100.0
            } else {
                0.0
            };
            let _ = app.emit(
                "punc-download-progress",
                serde_json::json!({ "downloaded": downloaded, "total": total_size, "percentage": pct }),
            );
            last_emit = Instant::now();
        }
    }

    // Final progress
    let _ = app.emit(
        "punc-download-progress",
        serde_json::json!({ "downloaded": downloaded, "total": total_size, "percentage": 100.0 }),
    );
    file.flush().map_err(|e| e.to_string())?;
    drop(file);

    // Extract .tar.bz2
    let archive_file = std::fs::File::open(&partial_path).map_err(|e| e.to_string())?;
    let bz2 = BzDecoder::new(archive_file);
    let mut tar = Archive::new(bz2);
    tar.unpack(&models_dir).map_err(|e| {
        let msg = format!("Failed to extract punc model archive: {}", e);
        let _ = app.emit("punc-download-failed", serde_json::json!({ "error": &msg }));
        msg
    })?;

    // Clean up partial file
    let _ = fs::remove_file(&partial_path);

    // Reset the punc model cache so the next transcription loads the freshly
    // downloaded model without requiring an app restart.
    crate::audio_toolkit::punc_zh::reset_cached_model();

    Ok(())
}
