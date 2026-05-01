use crate::managers::transcription::TranscriptionManager;
use crate::settings::default_asr_presets;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter, State};

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct BenchmarkItem {
    pub wav: String,
    pub hypothesis: String,
    pub latency_ms: u64,
    pub punc_count: usize,
    pub char_count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct BenchmarkSummary {
    pub total_items: usize,
    pub total_audio_seconds: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub punctuation_density: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct BenchmarkReport {
    pub preset_id: String,
    pub preset_name: String,
    pub model_id: String,
    pub language: String,
    pub punc_zh_enabled: bool,
    pub timestamp: String,
    pub items: Vec<BenchmarkItem>,
    pub summary: BenchmarkSummary,
}

#[derive(Serialize, Debug, Clone)]
struct BenchProgressPayload {
    preset_id: String,
    completed: usize,
    total: usize,
    current_wav: String,
}

/// Count punctuation characters in a string (。，、；：？！,.;:?!)
fn count_punc(s: &str) -> usize {
    s.chars()
        .filter(|c| {
            matches!(
                c,
                '。' | '，' | '、' | '；' | '：' | '？' | '！' | '.' | ',' | ';' | ':' | '?' | '!'
            )
        })
        .count()
}

/// Compute p-th percentile (0–100) from a sorted slice.
fn percentile(sorted: &[u64], p: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Run an ASR benchmark for a given preset over a directory of WAV files.
///
/// Steps:
/// 1. Apply the preset (loads model via existing `apply_asr_preset` logic).
/// 2. Wait for model to be ready (poll with timeout).
/// 3. Scan `dataset_dir` for *.wav files.
/// 4. Transcribe each file, measuring latency.
/// 5. Write a JSON report to `output_dir/{preset_id}_{timestamp}.json`.
/// 6. Return the `BenchmarkReport`.
///
/// Progress is emitted via the `bench-progress` event so the frontend (or
/// debug console) can track how many files have been processed.
#[tauri::command]
#[specta::specta]
pub async fn run_asr_benchmark(
    app: AppHandle,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
    dataset_dir: String,
    output_dir: String,
) -> Result<BenchmarkReport, String> {
    // ── Step 1: apply preset ────────────────────────────────────────────────
    let preset = default_asr_presets()
        .into_iter()
        .find(|p| p.id == preset_id)
        .ok_or_else(|| format!("Preset '{}' not found", preset_id))?;

    // Reuse the apply_asr_preset logic inline (avoids extra State parameter).
    {
        use crate::settings::{get_settings, write_settings};
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
        settings.selected_model = preset.model_id.clone();
        write_settings(&app, settings);
    }

    // Load model on blocking thread.
    let model_id_for_load = preset.model_id.clone();
    let tm_load = Arc::clone(&*transcription_manager);
    tauri::async_runtime::spawn_blocking(move || tm_load.load_model(&model_id_for_load))
        .await
        .map_err(|e| format!("Model load task panicked: {}", e))?
        .map_err(|e| format!("Failed to load preset model: {}", e))?;

    // ── Step 2: verify model is ready (poll up to 30 s via blocking thread) ─
    {
        let tm_check = Arc::clone(&*transcription_manager);
        tauri::async_runtime::spawn_blocking(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while !tm_check.is_model_loaded() {
                if std::time::Instant::now() >= deadline {
                    return Err("Timed out waiting for model to load".to_string());
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Ok(())
        })
        .await
        .map_err(|e| format!("Model readiness check panicked: {}", e))??;
    }

    // ── Step 3: collect WAV files ────────────────────────────────────────────
    let dataset_path = std::path::PathBuf::from(&dataset_dir);
    if !dataset_path.exists() {
        return Err(format!("dataset_dir does not exist: {}", dataset_dir));
    }

    let mut wav_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dataset_path)
        .map_err(|e| format!("Cannot read dataset_dir: {}", e))?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();

    if wav_paths.is_empty() {
        return Err(format!(
            "No WAV files found in dataset_dir: {}",
            dataset_dir
        ));
    }

    wav_paths.sort();
    let total = wav_paths.len();

    // ── Step 4: transcribe each file ────────────────────────────────────────
    let mut items: Vec<BenchmarkItem> = Vec::with_capacity(total);
    let mut total_audio_seconds = 0.0f64;

    // Determine language override from preset.
    let lang_override: Option<String> = if preset.language == "auto" {
        None
    } else {
        Some(preset.language.clone())
    };

    for (idx, wav_path) in wav_paths.iter().enumerate() {
        let wav_name = wav_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Emit progress event.
        let _ = app.emit(
            "bench-progress",
            BenchProgressPayload {
                preset_id: preset_id.clone(),
                completed: idx,
                total,
                current_wav: wav_name.clone(),
            },
        );

        // Read samples on blocking thread to avoid blocking async executor.
        let wav_path_clone = wav_path.clone();
        let samples = tauri::async_runtime::spawn_blocking(move || {
            crate::audio_toolkit::read_wav_samples(&wav_path_clone)
        })
        .await
        .map_err(|e| format!("WAV read task panicked for {}: {}", wav_name, e))?
        .map_err(|e| format!("Failed to read {}: {}", wav_name, e))?;

        // Estimate audio duration (16 kHz assumed).
        let audio_dur = samples.len() as f64 / 16000.0;
        total_audio_seconds += audio_dur;

        // Transcribe with timing.
        let tm_transcribe = Arc::clone(&*transcription_manager);
        let lang_clone = lang_override.clone();
        let wav_name_err = wav_name.clone();
        let t0 = Instant::now();
        let hypothesis = tauri::async_runtime::spawn_blocking(move || {
            tm_transcribe.transcribe_with_language_override(samples, lang_clone)
        })
        .await
        .map_err(|e| format!("Transcription task panicked for {}: {}", wav_name_err, e))?
        .unwrap_or_else(|e| {
            log::warn!("Transcription error for {}: {}", wav_name_err, e);
            String::new()
        });

        let latency_ms = t0.elapsed().as_millis() as u64;

        let punc_count = count_punc(&hypothesis);
        let char_count = hypothesis.chars().count();

        items.push(BenchmarkItem {
            wav: wav_name,
            hypothesis,
            latency_ms,
            punc_count,
            char_count,
        });
    }

    // Emit final progress.
    let _ = app.emit(
        "bench-progress",
        BenchProgressPayload {
            preset_id: preset_id.clone(),
            completed: total,
            total,
            current_wav: String::new(),
        },
    );

    // ── Step 5: compute summary ──────────────────────────────────────────────
    let mut latencies: Vec<u64> = items.iter().map(|i| i.latency_ms).collect();
    latencies.sort_unstable();

    let total_punc: usize = items.iter().map(|i| i.punc_count).sum();
    let total_chars: usize = items.iter().map(|i| i.char_count).sum();
    let punctuation_density = if total_chars > 0 {
        total_punc as f64 / total_chars as f64
    } else {
        0.0
    };

    let summary = BenchmarkSummary {
        total_items: items.len(),
        total_audio_seconds,
        p50_latency_ms: percentile(&latencies, 50),
        p95_latency_ms: percentile(&latencies, 95),
        punctuation_density,
    };

    // ── Step 6: write JSON report ────────────────────────────────────────────
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let report_filename = format!("{}_{}.json", preset_id, timestamp);

    let output_path = std::path::PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_path)
        .map_err(|e| format!("Cannot create output_dir '{}': {}", output_dir, e))?;

    let report = BenchmarkReport {
        preset_id: preset_id.clone(),
        preset_name: preset.name.clone(),
        model_id: preset.model_id.clone(),
        language: preset.language.clone(),
        punc_zh_enabled: preset.punc_zh_enabled,
        timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        items,
        summary,
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;

    let report_path = output_path.join(&report_filename);
    std::fs::write(&report_path, &json)
        .map_err(|e| format!("Failed to write report to {}: {}", report_path.display(), e))?;

    log::info!("Benchmark report written to: {}", report_path.display());

    Ok(report)
}
