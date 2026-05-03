use crate::cli::BenchMode;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::default_asr_presets;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter, State};

// ── Shared report types ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct BenchmarkItem {
    pub wav: String,
    pub hypothesis: String,
    pub latency_ms: u64,
    pub punc_count: usize,
    pub char_count: usize,
    // ── Accuracy mode extras ─────────────────────────────────────────────────
    /// Reference transcript (Accuracy mode only).
    #[serde(default)]
    pub reference: Option<String>,
    /// Character Error Rate = levenshtein(ref, hyp) / len(ref).  Accuracy mode only.
    #[serde(default)]
    pub cer: Option<f64>,
    /// Audio duration in milliseconds (Accuracy mode only; used to compute RTF).
    #[serde(default)]
    pub audio_ms: Option<u64>,
    /// Real-Time Factor = latency_ms / audio_ms (Accuracy mode only).
    #[serde(default)]
    pub rtf: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct BenchmarkSummary {
    /// P50 latency across all items (including cold-start first item).
    pub total_items: usize,
    pub total_audio_seconds: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub punctuation_density: f64,
    /// Latency of the very first item (cold-start: model warm-up, OnnxRuntime
    /// JIT compilation, CT-Punc first-load, etc.).  May be significantly higher
    /// than the steady-state numbers.  0 when total_items == 0.
    pub cold_start_latency_ms: u64,
    /// P50 latency of all items *except* the first (steady-state throughput).
    /// Reflects real-world interactive latency because the model stays loaded
    /// between utterances.  0 when total_items < 2.
    pub steady_p50_latency_ms: u64,
    /// P95 latency of all items *except* the first.
    /// 0 when total_items < 2.
    pub steady_p95_latency_ms: u64,
    // ── PuncOnly extras ──────────────────────────────────────────────────────
    /// P50 swap latency (BenchMode::Swap only). 0 for other modes.
    #[serde(default)]
    pub swap_p50_latency_ms: u64,
    /// P95 swap latency (BenchMode::Swap only). 0 for other modes.
    #[serde(default)]
    pub swap_p95_latency_ms: u64,
    // ── Accuracy mode extras ─────────────────────────────────────────────────
    /// Mean CER across all items (Accuracy mode only).
    #[serde(default)]
    pub mean_cer: f64,
    /// Median CER across all items (Accuracy mode only).
    #[serde(default)]
    pub median_cer: f64,
    /// P50 RTF across all items (Accuracy mode only).
    #[serde(default)]
    pub p50_rtf: f64,
    /// P95 RTF across all items (Accuracy mode only).
    #[serde(default)]
    pub p95_rtf: f64,
    /// P99 RTF across all items (Accuracy mode only).
    #[serde(default)]
    pub p99_rtf: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct BenchmarkReport {
    pub preset_id: String,
    pub preset_name: String,
    pub model_id: String,
    pub language: String,
    pub punc_zh_enabled: bool,
    pub timestamp: String,
    /// Benchmark mode that produced this report.
    #[serde(default = "default_bench_mode_str")]
    pub bench_mode: String,
    pub items: Vec<BenchmarkItem>,
    pub summary: BenchmarkSummary,
    // ── Chain mode extras ────────────────────────────────────────────────────
    /// Ordered chain step labels (BenchMode::Chain only).
    #[serde(default)]
    pub chain_steps: Vec<ChainStepResult>,
    // ── Swap mode extras ─────────────────────────────────────────────────────
    /// Per-swap latency records (BenchMode::Swap only).
    #[serde(default)]
    pub swap_records: Vec<SwapRecord>,
}

fn default_bench_mode_str() -> String {
    "asr".to_string()
}

/// One step in a Chain-mode run.
#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct ChainStepResult {
    pub step: usize,
    pub prompt_id: String,
    pub input_len: usize,
    pub output_len: usize,
    pub latency_ms: u64,
    pub output_preview: String,
}

/// One hot-swap measurement (unload + load of one engine).
#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct SwapRecord {
    pub swap_index: usize,
    pub from_model: String,
    pub to_model: String,
    pub unload_ms: u64,
    pub load_ms: u64,
    pub total_ms: u64,
}

// ── Progress payload (shared by all modes) ───────────────────────────────────

#[derive(Serialize, Debug, Clone)]
struct BenchProgressPayload {
    preset_id: String,
    completed: usize,
    total: usize,
    current_wav: String,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

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

fn make_summary(items: &[BenchmarkItem], total_audio_seconds: f64) -> BenchmarkSummary {
    let mut latencies: Vec<u64> = items.iter().map(|i| i.latency_ms).collect();
    latencies.sort_unstable();

    let total_punc: usize = items.iter().map(|i| i.punc_count).sum();
    let total_chars: usize = items.iter().map(|i| i.char_count).sum();
    let punctuation_density = if total_chars > 0 {
        total_punc as f64 / total_chars as f64
    } else {
        0.0
    };

    let cold_start_latency_ms = items.first().map(|i| i.latency_ms).unwrap_or(0);
    let mut steady_latencies: Vec<u64> = items.iter().skip(1).map(|i| i.latency_ms).collect();
    steady_latencies.sort_unstable();

    BenchmarkSummary {
        total_items: items.len(),
        total_audio_seconds,
        p50_latency_ms: percentile(&latencies, 50),
        p95_latency_ms: percentile(&latencies, 95),
        punctuation_density,
        cold_start_latency_ms,
        steady_p50_latency_ms: percentile(&steady_latencies, 50),
        steady_p95_latency_ms: percentile(&steady_latencies, 95),
        swap_p50_latency_ms: 0,
        swap_p95_latency_ms: 0,
        mean_cer: 0.0,
        median_cer: 0.0,
        p50_rtf: 0.0,
        p95_rtf: 0.0,
        p99_rtf: 0.0,
    }
}

// ── Main Tauri command ────────────────────────────────────────────────────────

/// Run a benchmark. `mode` selects the pipeline to exercise:
///
/// * `BenchMode::Asr`      — original WAV→transcribe path (default, unchanged)
/// * `BenchMode::PuncOnly` — punctuation-only loop, 100 iterations
/// * `BenchMode::Chain`    — post-process chain timing
/// * `BenchMode::Swap`     — engine hot-swap lifecycle timing
/// * `BenchMode::Accuracy` — CER + RTF evaluation against a reference manifest
///
/// The `preset_id`, `dataset_dir`, and `output_dir` parameters remain unchanged
/// so the existing ASR path and CLI dispatcher are fully backward-compatible.
#[tauri::command]
#[specta::specta]
pub async fn run_asr_benchmark(
    app: AppHandle,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
    dataset_dir: String,
    output_dir: String,
    #[allow(unused_variables)] mode: Option<String>,
) -> Result<BenchmarkReport, String> {
    // Parse mode string (None or missing → Asr for backward-compat).
    let bench_mode = match mode.as_deref() {
        // Accept both hyphenated (clap CLI output: "punc-only") and underscore
        // forms (camelCase legacy) so that the single-instance forwarder works
        // regardless of which serialisation clap produces.
        Some("punc-only") | Some("punc_only") | Some("PuncOnly") => BenchMode::PuncOnly,
        Some("chain") | Some("Chain") => BenchMode::Chain,
        Some("swap") | Some("Swap") => BenchMode::Swap,
        Some("accuracy") | Some("Accuracy") => BenchMode::Accuracy,
        _ => BenchMode::Asr,
    };

    match bench_mode {
        BenchMode::Asr => {
            run_asr_mode(
                app,
                transcription_manager,
                preset_id,
                dataset_dir,
                output_dir,
            )
            .await
        }
        BenchMode::PuncOnly => {
            run_punc_only_mode(
                app,
                transcription_manager,
                preset_id,
                dataset_dir,
                output_dir,
            )
            .await
        }
        BenchMode::Chain => {
            run_chain_mode(
                app,
                transcription_manager,
                preset_id,
                dataset_dir,
                output_dir,
            )
            .await
        }
        BenchMode::Swap => {
            run_swap_mode(
                app,
                transcription_manager,
                preset_id,
                dataset_dir,
                output_dir,
            )
            .await
        }
        BenchMode::Accuracy => {
            run_accuracy_mode(
                app,
                transcription_manager,
                preset_id,
                dataset_dir,
                output_dir,
            )
            .await
        }
    }
}

// ── BenchMode::Asr ───────────────────────────────────────────────────────────

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
async fn run_asr_mode(
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
    //
    // Performance note: the entire WAV-read + transcribe loop runs inside a
    // *single* spawn_blocking task so that:
    //   a) per-item async/thread-pool dispatch overhead (~5–50 ms) is eliminated
    //      from the timing measurements;
    //   b) the engine mutex is never dropped between items, so the OS does not
    //      have to context-switch away from the worker thread mid-loop.
    //
    // Progress events are emitted from inside the blocking thread via the
    // app handle (which is Send + Sync).

    // Determine language override from preset.
    let lang_override: Option<String> = if preset.language == "auto" {
        None
    } else {
        Some(preset.language.clone())
    };

    let tm_bench = Arc::clone(&*transcription_manager);
    let app_bench = app.clone();
    let preset_id_bench = preset_id.clone();
    let lang_override_bench = lang_override.clone();

    let (items, total_audio_seconds): (Vec<BenchmarkItem>, f64) =
        tauri::async_runtime::spawn_blocking(move || {
            let mut items: Vec<BenchmarkItem> = Vec::with_capacity(total);
            let mut total_audio_seconds = 0.0f64;

            for (idx, wav_path) in wav_paths.iter().enumerate() {
                let wav_name = wav_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Emit progress event (fire-and-forget; bench continues even if emit fails).
                let _ = app_bench.emit(
                    "bench-progress",
                    BenchProgressPayload {
                        preset_id: preset_id_bench.clone(),
                        completed: idx,
                        total,
                        current_wav: wav_name.clone(),
                    },
                );

                // Read WAV samples synchronously — we are already on a blocking thread.
                let samples = match crate::audio_toolkit::read_wav_samples(wav_path) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("Failed to read {}: {}; skipping", wav_name, e);
                        continue;
                    }
                };

                // Estimate audio duration (16 kHz assumed).
                let audio_dur = samples.len() as f64 / 16000.0;
                total_audio_seconds += audio_dur;

                // Transcribe with tight timing — only the engine call is timed.
                let t0 = Instant::now();
                let hypothesis = tm_bench
                    .transcribe_with_language_override(samples, lang_override_bench.clone())
                    .unwrap_or_else(|e| {
                        log::warn!("Transcription error for {}: {}", wav_name, e);
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
                    reference: None,
                    cer: None,
                    audio_ms: None,
                    rtf: None,
                });
            }

            (items, total_audio_seconds)
        })
        .await
        .map_err(|e| format!("Benchmark loop task panicked: {}", e))?;

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
    let summary = make_summary(&items, total_audio_seconds);

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
        bench_mode: "asr".to_string(),
        items,
        summary,
        chain_steps: vec![],
        swap_records: vec![],
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;

    let report_path = output_path.join(&report_filename);
    std::fs::write(&report_path, &json)
        .map_err(|e| format!("Failed to write report to {}: {}", report_path.display(), e))?;

    log::info!("Benchmark report written to: {}", report_path.display());

    Ok(report)
}

// ── BenchMode::PuncOnly ───────────────────────────────────────────────────────

/// Run the punctuation-only benchmark.
///
/// Reads `<dataset_dir>/punc_input.txt` (one plain-text Chinese sentence per
/// line).  Each line is fed to `punc_zh::add_punctuation` in a loop of 100
/// iterations.  The first iteration constitutes the cold-start; iterations 2-100
/// build the steady-state distribution.
async fn run_punc_only_mode(
    app: AppHandle,
    _transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
    dataset_dir: String,
    output_dir: String,
) -> Result<BenchmarkReport, String> {
    // Resolve punc model dir.
    let model_dir = crate::portable::app_data_dir(&app)
        .map_err(|e| format!("Cannot resolve app_data_dir: {}", e))?
        .join("models")
        .join("sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8");

    if !crate::audio_toolkit::punc_zh::is_punc_model_present(&model_dir) {
        return Err(format!(
            "Punc model not found at {}. Download via the app settings first.",
            model_dir.display()
        ));
    }

    // Read input sentences.
    let input_file = std::path::PathBuf::from(&dataset_dir).join("punc_input.txt");
    if !input_file.exists() {
        return Err(format!(
            "punc_input.txt not found at {}",
            input_file.display()
        ));
    }

    let content = std::fs::read_to_string(&input_file)
        .map_err(|e| format!("Cannot read punc_input.txt: {}", e))?;

    let sentences: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if sentences.is_empty() {
        return Err("punc_input.txt contains no non-empty lines".to_string());
    }

    const ITERATIONS: usize = 100;
    let total_sentences = sentences.len();

    // Run on a blocking thread to avoid stalling the async runtime.
    let model_dir_clone = model_dir.clone();
    let sentences_owned: Vec<String> = sentences.iter().map(|s| s.to_string()).collect();
    let preset_id_clone = preset_id.clone();
    let app_clone = app.clone();

    let items: Vec<BenchmarkItem> = tauri::async_runtime::spawn_blocking(move || {
        let mut items: Vec<BenchmarkItem> = Vec::with_capacity(ITERATIONS * total_sentences);

        for iter in 0..ITERATIONS {
            for (sent_idx, sentence) in sentences_owned.iter().enumerate() {
                // Emit progress.
                let completed = iter * total_sentences + sent_idx;
                let _ = app_clone.emit(
                    "bench-progress",
                    BenchProgressPayload {
                        preset_id: preset_id_clone.clone(),
                        completed,
                        total: ITERATIONS * total_sentences,
                        current_wav: format!("iter={} sent={}", iter, sent_idx),
                    },
                );

                let t0 = Instant::now();
                let result =
                    crate::audio_toolkit::punc_zh::add_punctuation(&model_dir_clone, sentence)
                        .unwrap_or_else(|e| {
                            log::warn!("punc_zh error at iter={} sent={}: {}", iter, sent_idx, e);
                            sentence.to_string()
                        });
                let latency_ms = t0.elapsed().as_millis() as u64;

                let punc_count = count_punc(&result);
                let char_count = result.chars().count();

                items.push(BenchmarkItem {
                    wav: format!("iter{:03}_sent{:02}", iter, sent_idx),
                    hypothesis: result,
                    latency_ms,
                    punc_count,
                    char_count,
                    reference: None,
                    cer: None,
                    audio_ms: None,
                    rtf: None,
                });
            }
        }

        items
    })
    .await
    .map_err(|e| format!("PuncOnly bench task panicked: {}", e))?;

    let summary = make_summary(&items, 0.0);

    // Write report.
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let report_filename = format!("{}_punc_only_{}.json", preset_id, timestamp);
    let output_path = std::path::PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_path)
        .map_err(|e| format!("Cannot create output_dir '{}': {}", output_dir, e))?;

    let report = BenchmarkReport {
        preset_id: preset_id.clone(),
        preset_name: "PuncOnly".to_string(),
        model_id: model_dir.display().to_string(),
        language: "zh".to_string(),
        punc_zh_enabled: true,
        timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        bench_mode: "punc_only".to_string(),
        items,
        summary,
        chain_steps: vec![],
        swap_records: vec![],
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;
    let report_path = output_path.join(&report_filename);
    std::fs::write(&report_path, &json).map_err(|e| format!("Failed to write report: {}", e))?;

    log::info!(
        "PuncOnly benchmark report written to: {}",
        report_path.display()
    );

    Ok(report)
}

// ── BenchMode::Chain ──────────────────────────────────────────────────────────

/// Chain-test input format (JSON).
#[derive(Deserialize, Debug)]
struct ChainTestInput {
    input: String,
    chain: Vec<String>,
}

/// Run the post-process chain benchmark.
///
/// Reads `<dataset_dir>/chain_test.json`:
/// ```json
/// { "input": "今天天气很好", "chain": ["prompt_id_1", "prompt_id_2"] }
/// ```
///
/// All prompt IDs must already exist in `settings.post_process_prompts`.
/// Each chain step is timed individually; total latency is also recorded.
async fn run_chain_mode(
    app: AppHandle,
    _transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
    dataset_dir: String,
    output_dir: String,
) -> Result<BenchmarkReport, String> {
    // Read chain_test.json.
    let chain_file = std::path::PathBuf::from(&dataset_dir).join("chain_test.json");
    if !chain_file.exists() {
        return Err(format!(
            "chain_test.json not found at {}",
            chain_file.display()
        ));
    }

    let content = std::fs::read_to_string(&chain_file)
        .map_err(|e| format!("Cannot read chain_test.json: {}", e))?;
    let chain_input: ChainTestInput =
        serde_json::from_str(&content).map_err(|e| format!("Invalid chain_test.json: {}", e))?;

    if chain_input.chain.is_empty() {
        return Err("chain_test.json: 'chain' array must not be empty".to_string());
    }

    // Load current settings and validate that all prompt IDs exist.
    let settings = crate::settings::get_settings(&app);

    // Verify all prompt IDs exist before running.
    let missing: Vec<&str> = chain_input
        .chain
        .iter()
        .filter(|id| !settings.post_process_prompts.iter().any(|p| &p.id == *id))
        .map(|id| id.as_str())
        .collect();

    if !missing.is_empty() {
        let available: Vec<String> = settings
            .post_process_prompts
            .iter()
            .map(|p| format!("'{}' ({})", p.id, p.name))
            .collect();
        return Err(format!(
            "Chain prompt IDs not found: [{}]. Available prompts: [{}]",
            missing.join(", "),
            available.join(", ")
        ));
    }

    // Build a temporary settings object that uses our chain.
    let mut chain_settings = settings.clone();
    chain_settings.post_process_chain = Some(chain_input.chain.clone());

    // Run chain steps individually for per-step timing.
    let mut chain_steps: Vec<ChainStepResult> = Vec::new();
    let mut current_text = chain_input.input.clone();
    let overall_t0 = Instant::now();

    for (step_idx, prompt_id) in chain_input.chain.iter().enumerate() {
        let prompt_template = settings
            .post_process_prompts
            .iter()
            .find(|p| &p.id == prompt_id)
            .map(|p| p.prompt.clone())
            .unwrap(); // validated above

        // Build a one-step settings override (so we always measure one step at a time).
        let mut step_settings = settings.clone();
        step_settings.post_process_chain = Some(vec![prompt_id.clone()]);

        let input_text = current_text.clone();
        let step_t0 = Instant::now();

        let result =
            crate::actions::post_process_transcription(&step_settings, &input_text, None).await;

        let step_latency_ms = step_t0.elapsed().as_millis() as u64;

        let output_text = result.unwrap_or_else(|| {
            log::warn!(
                "Chain step {} ('{}') returned None; keeping previous text",
                step_idx + 1,
                prompt_id
            );
            input_text.clone()
        });

        let preview_len = output_text.len().min(200);
        chain_steps.push(ChainStepResult {
            step: step_idx + 1,
            prompt_id: prompt_id.clone(),
            input_len: input_text.chars().count(),
            output_len: output_text.chars().count(),
            latency_ms: step_latency_ms,
            output_preview: output_text[..preview_len].to_string(),
        });

        // Emit progress.
        let _ = app.emit(
            "bench-progress",
            BenchProgressPayload {
                preset_id: preset_id.clone(),
                completed: step_idx + 1,
                total: chain_input.chain.len(),
                current_wav: format!("step={} prompt={}", step_idx + 1, prompt_id),
            },
        );

        // Log prompt template for debugging (truncated).
        let preview = if prompt_template.len() > 80 {
            format!("{}...", &prompt_template[..80])
        } else {
            prompt_template.clone()
        };
        log::debug!(
            "Chain step {} ('{}') latency={}ms prompt='{}'",
            step_idx + 1,
            prompt_id,
            step_latency_ms,
            preview
        );

        current_text = output_text;
    }

    let total_latency_ms = overall_t0.elapsed().as_millis() as u64;

    // Build a synthetic BenchmarkItem to represent the full chain run.
    let total_latencies: Vec<u64> = chain_steps.iter().map(|s| s.latency_ms).collect();
    let punc_count = count_punc(&current_text);
    let char_count = current_text.chars().count();

    let items = vec![BenchmarkItem {
        wav: "chain_run".to_string(),
        hypothesis: current_text.clone(),
        latency_ms: total_latency_ms,
        punc_count,
        char_count,
        reference: None,
        cer: None,
        audio_ms: None,
        rtf: None,
    }];

    // Build per-step summary metrics.
    let mut sorted_step_latencies = total_latencies.clone();
    sorted_step_latencies.sort_unstable();

    let summary = BenchmarkSummary {
        total_items: chain_steps.len(),
        total_audio_seconds: 0.0,
        p50_latency_ms: percentile(&sorted_step_latencies, 50),
        p95_latency_ms: percentile(&sorted_step_latencies, 95),
        punctuation_density: if char_count > 0 {
            punc_count as f64 / char_count as f64
        } else {
            0.0
        },
        cold_start_latency_ms: chain_steps.first().map(|s| s.latency_ms).unwrap_or(0),
        steady_p50_latency_ms: {
            let mut rest: Vec<u64> = chain_steps.iter().skip(1).map(|s| s.latency_ms).collect();
            rest.sort_unstable();
            percentile(&rest, 50)
        },
        steady_p95_latency_ms: {
            let mut rest: Vec<u64> = chain_steps.iter().skip(1).map(|s| s.latency_ms).collect();
            rest.sort_unstable();
            percentile(&rest, 95)
        },
        swap_p50_latency_ms: 0,
        swap_p95_latency_ms: 0,
        mean_cer: 0.0,
        median_cer: 0.0,
        p50_rtf: 0.0,
        p95_rtf: 0.0,
        p99_rtf: 0.0,
    };

    // Write report.
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let report_filename = format!("{}_chain_{}.json", preset_id, timestamp);
    let output_path = std::path::PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_path)
        .map_err(|e| format!("Cannot create output_dir '{}': {}", output_dir, e))?;

    let report = BenchmarkReport {
        preset_id: preset_id.clone(),
        preset_name: "Chain".to_string(),
        model_id: "n/a".to_string(),
        language: "n/a".to_string(),
        punc_zh_enabled: false,
        timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        bench_mode: "chain".to_string(),
        items,
        summary,
        chain_steps,
        swap_records: vec![],
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;
    let report_path = output_path.join(&report_filename);
    std::fs::write(&report_path, &json).map_err(|e| format!("Failed to write report: {}", e))?;

    log::info!(
        "Chain benchmark report written to: {}",
        report_path.display()
    );

    Ok(report)
}

// ── BenchMode::Swap ───────────────────────────────────────────────────────────

/// Run the engine hot-swap benchmark.
///
/// Executes the sequence [sense-voice-int8 → funasr-nano → sense-voice-int8] × 3
/// (9 swap operations).  Each operation records `unload_ms`, `load_ms`, and
/// `total_ms`.  The summary contains `swap_p50_latency_ms` / `swap_p95_latency_ms`
/// (total latency per swap), plus the usual cold/steady split (first swap = cold).
///
/// No actual transcription is performed — only model lifecycle is tested.
async fn run_swap_mode(
    app: AppHandle,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
    _dataset_dir: String,
    output_dir: String,
) -> Result<BenchmarkReport, String> {
    // The swap sequence: alternating between two models, 3 full round-trips = 9 swaps.
    let sequence = [
        "sense-voice-int8",
        "funasr-nano",
        "sense-voice-int8",
        "funasr-nano",
        "sense-voice-int8",
        "funasr-nano",
        "sense-voice-int8",
        "funasr-nano",
        "sense-voice-int8",
    ];

    let total_swaps = sequence.len(); // 9
    let mut swap_records: Vec<SwapRecord> = Vec::with_capacity(total_swaps);

    // Ensure we start unloaded so the first load is always a cold swap.
    {
        let tm = Arc::clone(&*transcription_manager);
        if tm.is_model_loaded() {
            tm.unload_model()
                .map_err(|e| format!("Failed to unload model before swap bench: {}", e))?;
        }
    }

    let from_model_tracker: std::sync::Mutex<String> = std::sync::Mutex::new("(none)".to_string());

    for (swap_idx, &to_model) in sequence.iter().enumerate() {
        let from_model = from_model_tracker.lock().unwrap().clone();

        // Emit progress.
        let _ = app.emit(
            "bench-progress",
            BenchProgressPayload {
                preset_id: preset_id.clone(),
                completed: swap_idx,
                total: total_swaps,
                current_wav: format!("swap {}: {} → {}", swap_idx + 1, from_model, to_model),
            },
        );

        let tm_swap = Arc::clone(&*transcription_manager);
        let to_model_str = to_model.to_string();
        let from_model_str = from_model.clone();

        let record: SwapRecord = tauri::async_runtime::spawn_blocking(move || {
            // Unload (only if something is loaded).
            let unload_t0 = Instant::now();
            if tm_swap.is_model_loaded() {
                tm_swap
                    .unload_model()
                    .unwrap_or_else(|e| log::warn!("Swap unload error: {}", e));
            }
            let unload_ms = unload_t0.elapsed().as_millis() as u64;

            // Load target model.
            let load_t0 = Instant::now();
            if let Err(e) = tm_swap.load_model(&to_model_str) {
                log::warn!("Swap load error for {}: {}", to_model_str, e);
            }
            let load_ms = load_t0.elapsed().as_millis() as u64;

            SwapRecord {
                swap_index: swap_idx + 1,
                from_model: from_model_str,
                to_model: to_model_str,
                unload_ms,
                load_ms,
                total_ms: unload_ms + load_ms,
            }
        })
        .await
        .map_err(|e| format!("Swap bench task panicked at swap {}: {}", swap_idx + 1, e))?;

        log::info!(
            "Swap {}/{}: {} → {} — unload={}ms load={}ms total={}ms",
            swap_idx + 1,
            total_swaps,
            record.from_model,
            record.to_model,
            record.unload_ms,
            record.load_ms,
            record.total_ms
        );

        *from_model_tracker.lock().unwrap() = to_model.to_string();
        swap_records.push(record);
    }

    // Emit final progress.
    let _ = app.emit(
        "bench-progress",
        BenchProgressPayload {
            preset_id: preset_id.clone(),
            completed: total_swaps,
            total: total_swaps,
            current_wav: String::new(),
        },
    );

    // Build synthetic BenchmarkItem list from swap records.
    let items: Vec<BenchmarkItem> = swap_records
        .iter()
        .map(|r| BenchmarkItem {
            wav: format!(
                "swap_{:02}_{}_to_{}",
                r.swap_index, r.from_model, r.to_model
            ),
            hypothesis: String::new(),
            latency_ms: r.total_ms,
            punc_count: 0,
            char_count: 0,
            reference: None,
            cer: None,
            audio_ms: None,
            rtf: None,
        })
        .collect();

    // Swap-specific percentiles.
    let mut swap_totals: Vec<u64> = swap_records.iter().map(|r| r.total_ms).collect();
    swap_totals.sort_unstable();

    let cold_start_latency_ms = swap_records.first().map(|r| r.total_ms).unwrap_or(0);
    let mut steady_totals: Vec<u64> = swap_records.iter().skip(1).map(|r| r.total_ms).collect();
    steady_totals.sort_unstable();

    let summary = BenchmarkSummary {
        total_items: swap_records.len(),
        total_audio_seconds: 0.0,
        p50_latency_ms: percentile(&swap_totals, 50),
        p95_latency_ms: percentile(&swap_totals, 95),
        punctuation_density: 0.0,
        cold_start_latency_ms,
        steady_p50_latency_ms: percentile(&steady_totals, 50),
        steady_p95_latency_ms: percentile(&steady_totals, 95),
        swap_p50_latency_ms: percentile(&swap_totals, 50),
        swap_p95_latency_ms: percentile(&swap_totals, 95),
        mean_cer: 0.0,
        median_cer: 0.0,
        p50_rtf: 0.0,
        p95_rtf: 0.0,
        p99_rtf: 0.0,
    };

    // Write report.
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let report_filename = format!("{}_swap_{}.json", preset_id, timestamp);
    let output_path = std::path::PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_path)
        .map_err(|e| format!("Cannot create output_dir '{}': {}", output_dir, e))?;

    let report = BenchmarkReport {
        preset_id: preset_id.clone(),
        preset_name: "Swap".to_string(),
        model_id: "sense-voice-int8 / funasr-nano".to_string(),
        language: "n/a".to_string(),
        punc_zh_enabled: false,
        timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        bench_mode: "swap".to_string(),
        items,
        summary,
        chain_steps: vec![],
        swap_records,
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;
    let report_path = output_path.join(&report_filename);
    std::fs::write(&report_path, &json).map_err(|e| format!("Failed to write report: {}", e))?;

    log::info!(
        "Swap benchmark report written to: {}",
        report_path.display()
    );

    Ok(report)
}

// ── BenchMode::Accuracy ───────────────────────────────────────────────────────

/// One entry from the manifest.jsonl corpus file.
#[derive(Deserialize, Debug)]
struct ManifestEntry {
    wav: String,
    #[serde(rename = "ref")]
    reference: String,
    #[serde(default)]
    tags: Vec<String>,
}

/// Compute f64 percentile (0–100) from a sorted slice of f64 values.
fn percentile_f64(sorted: &[f64], p: u8) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Run the accuracy benchmark.
///
/// Reads `<dataset_dir>/manifest.jsonl` (one JSON object per line):
/// ```json
/// {"wav": "01_pure_zh.wav", "ref": "你好世界", "tags": ["pure_zh"]}
/// ```
///
/// For each entry the function:
/// 1. Reads the WAV file with `read_wav_samples` (assumes 16 kHz mono).
/// 2. Transcribes via the loaded model.
/// 3. Computes CER and RTF.
///
/// Aggregate statistics (mean/median CER, p50/p95/p99 RTF) are stored in the
/// `BenchmarkSummary` extension fields added for accuracy mode.
///
/// The preset is applied exactly the same way as in `run_asr_mode` so the
/// engine under test is fully configured before transcription begins.
async fn run_accuracy_mode(
    app: AppHandle,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    preset_id: String,
    dataset_dir: String,
    output_dir: String,
) -> Result<BenchmarkReport, String> {
    // ── Step 1: apply preset (same logic as run_asr_mode) ──────────────────
    let preset = crate::settings::default_asr_presets()
        .into_iter()
        .find(|p| p.id == preset_id)
        .ok_or_else(|| format!("Preset '{}' not found", preset_id))?;

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

    let model_id_for_load = preset.model_id.clone();
    let tm_load = Arc::clone(&*transcription_manager);
    tauri::async_runtime::spawn_blocking(move || tm_load.load_model(&model_id_for_load))
        .await
        .map_err(|e| format!("Model load task panicked: {}", e))?
        .map_err(|e| format!("Failed to load preset model: {}", e))?;

    // Wait for model to be ready (poll up to 30 s).
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

    // ── Step 2: read manifest.jsonl ─────────────────────────────────────────
    let dataset_path = std::path::PathBuf::from(&dataset_dir);
    let manifest_path = dataset_path.join("manifest.jsonl");
    if !manifest_path.exists() {
        return Err(format!(
            "manifest.jsonl not found at {}",
            manifest_path.display()
        ));
    }

    let manifest_content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Cannot read manifest.jsonl: {}", e))?;

    let entries: Vec<ManifestEntry> = manifest_content
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_no, line)| {
            serde_json::from_str::<ManifestEntry>(line).map_err(|e| {
                format!("manifest.jsonl line {}: parse error: {}", line_no + 1, e)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if entries.is_empty() {
        return Err("manifest.jsonl contains no entries".to_string());
    }

    let total = entries.len();

    // ── Step 3: transcribe each entry on a blocking thread ─────────────────
    let lang_override: Option<String> = if preset.language == "auto" {
        None
    } else {
        Some(preset.language.clone())
    };

    let tm_bench = Arc::clone(&*transcription_manager);
    let app_bench = app.clone();
    let preset_id_bench = preset_id.clone();
    let lang_override_bench = lang_override.clone();

    let items: Vec<BenchmarkItem> = tauri::async_runtime::spawn_blocking(move || {
        let mut items: Vec<BenchmarkItem> = Vec::with_capacity(total);

        for (idx, entry) in entries.iter().enumerate() {
            let wav_name = entry.wav.clone();

            // Emit progress.
            let _ = app_bench.emit(
                "bench-progress",
                BenchProgressPayload {
                    preset_id: preset_id_bench.clone(),
                    completed: idx,
                    total,
                    current_wav: wav_name.clone(),
                },
            );

            // Resolve WAV path relative to the manifest directory.
            let wav_path = dataset_path.join(&wav_name);

            // Read samples (16 kHz mono assumed; format matches fixture generation).
            let samples = match crate::audio_toolkit::read_wav_samples(&wav_path) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!(
                        "Accuracy bench: failed to read {}: {}; skipping",
                        wav_name,
                        e
                    );
                    continue;
                }
            };

            // Audio duration in milliseconds (assumes 16 kHz).
            let audio_ms = (samples.len() as f64 / 16000.0 * 1000.0) as u64;

            // Transcribe.
            let t0 = Instant::now();
            let hypothesis = tm_bench
                .transcribe_with_language_override(samples, lang_override_bench.clone())
                .unwrap_or_else(|e| {
                    log::warn!("Accuracy bench: transcription error for {}: {}", wav_name, e);
                    String::new()
                });
            let latency_ms = t0.elapsed().as_millis() as u64;

            // CER and RTF.
            let cer = crate::audio_toolkit::cer::character_error_rate(&entry.reference, &hypothesis);
            let rtf = if audio_ms > 0 {
                latency_ms as f64 / audio_ms as f64
            } else {
                0.0
            };

            let punc_count = count_punc(&hypothesis);
            let char_count = hypothesis.chars().count();

            items.push(BenchmarkItem {
                wav: wav_name,
                hypothesis,
                latency_ms,
                punc_count,
                char_count,
                reference: Some(entry.reference.clone()),
                cer: Some(cer),
                audio_ms: Some(audio_ms),
                rtf: Some(rtf),
            });

            log::debug!(
                "Accuracy bench [{}/{}] {} — latency={}ms audio={}ms RTF={:.3} CER={:.3}",
                idx + 1,
                total,
                entry.wav,
                latency_ms,
                audio_ms,
                rtf,
                cer
            );

            // Log tag info for filtered analysis.
            if !entry.tags.is_empty() {
                log::debug!("  tags: {}", entry.tags.join(", "));
            }
        }

        items
    })
    .await
    .map_err(|e| format!("Accuracy bench loop task panicked: {}", e))?;

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

    // ── Step 4: aggregate statistics ────────────────────────────────────────

    // Collect CER and RTF values for items that were successfully processed.
    let mut cer_values: Vec<f64> = items
        .iter()
        .filter_map(|it| it.cer)
        .collect();
    let mut rtf_values: Vec<f64> = items
        .iter()
        .filter_map(|it| it.rtf)
        .collect();
    cer_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    rtf_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_cer = if cer_values.is_empty() {
        0.0
    } else {
        cer_values.iter().sum::<f64>() / cer_values.len() as f64
    };
    let median_cer = percentile_f64(&cer_values, 50);
    let p50_rtf = percentile_f64(&rtf_values, 50);
    let p95_rtf = percentile_f64(&rtf_values, 95);
    let p99_rtf = percentile_f64(&rtf_values, 99);

    // Build the core summary using the shared helper, then patch accuracy fields.
    let total_audio_seconds: f64 = items
        .iter()
        .filter_map(|it| it.audio_ms)
        .map(|ms| ms as f64 / 1000.0)
        .sum();

    let mut summary = make_summary(&items, total_audio_seconds);
    summary.mean_cer = mean_cer;
    summary.median_cer = median_cer;
    summary.p50_rtf = p50_rtf;
    summary.p95_rtf = p95_rtf;
    summary.p99_rtf = p99_rtf;

    log::info!(
        "Accuracy bench summary — items={} mean_CER={:.4} median_CER={:.4} \
         p50_RTF={:.3} p95_RTF={:.3} p99_RTF={:.3}",
        items.len(),
        mean_cer,
        median_cer,
        p50_rtf,
        p95_rtf,
        p99_rtf,
    );

    // ── Step 5: write JSON report ────────────────────────────────────────────
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let report_filename = format!("{}_accuracy_{}.json", preset_id, timestamp);

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
        bench_mode: "accuracy".to_string(),
        items,
        summary,
        chain_steps: vec![],
        swap_records: vec![],
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;

    let report_path = output_path.join(&report_filename);
    std::fs::write(&report_path, &json)
        .map_err(|e| format!("Failed to write report to {}: {}", report_path.display(), e))?;

    log::info!(
        "Accuracy benchmark report written to: {}",
        report_path.display()
    );

    Ok(report)
}

// ── BenchMode::Accuracy (in-progress) ───────────────────────────────────────
