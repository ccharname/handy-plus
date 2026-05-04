use clap::{Parser, ValueEnum};

/// Selects which benchmark mode to run when `--bench-preset` is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum BenchMode {
    /// WAV → ASR transcription (default, existing behaviour).
    #[default]
    Asr,
    /// Punctuation-only: feed plain text through `punc_zh::add_punctuation` 100×.
    /// Dataset: `<bench-dataset>/punc_input.txt` (one sentence per line, ≥10 lines).
    PuncOnly,
    /// Post-process chain: run `post_process_transcription` on a test input.
    /// Dataset: `<bench-dataset>/chain_test.json`
    /// `{ "input": "...", "chain": ["prompt_id_1", "prompt_id_2"] }`
    Chain,
    /// Engine hot-swap: [sense-voice-int8 → funasr-nano → sense-voice-int8] × 3.
    /// Measures unload+load lifecycle time, no actual transcription.
    Swap,
    /// Accuracy benchmark: transcribe a manifest of reference-labelled WAV files
    /// and compute per-item CER + RTF, plus aggregate mean/median CER and
    /// p50/p95/p99 RTF.  Dataset: `<bench-dataset>/manifest.jsonl`.
    Accuracy,
}

#[derive(Parser, Debug, Clone, Default)]
#[command(name = "handy", about = "Handy - Speech to Text")]
pub struct CliArgs {
    /// Start with the main window hidden
    #[arg(long)]
    pub start_hidden: bool,

    /// Disable the system tray icon
    #[arg(long)]
    pub no_tray: bool,

    /// Toggle transcription on/off (sent to running instance)
    #[arg(long)]
    pub toggle_transcription: bool,

    /// Toggle transcription with post-processing on/off (sent to running instance)
    #[arg(long)]
    pub toggle_post_process: bool,

    /// Cancel the current operation (sent to running instance)
    #[arg(long)]
    pub cancel: bool,

    /// Enable debug mode with verbose logging
    #[arg(long)]
    pub debug: bool,

    /// Include full transcript text in observability logs (default: off for privacy)
    #[arg(long)]
    pub log_transcripts: bool,

    /// Run ASR benchmark for a preset (sent to running instance); requires --bench-dataset and --bench-output
    #[arg(long)]
    pub bench_preset: Option<String>,

    /// Dataset directory containing WAV files for benchmark
    #[arg(long)]
    pub bench_dataset: Option<String>,

    /// Output directory for benchmark JSON reports
    #[arg(long)]
    pub bench_output: Option<String>,

    /// Benchmark mode (default: asr). Selects which pipeline to exercise.
    #[arg(long, value_enum, default_value_t = BenchMode::Asr)]
    pub bench_mode: BenchMode,
}
