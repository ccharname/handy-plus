use clap::Parser;

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

    /// Run ASR benchmark for a preset (sent to running instance); requires --bench-dataset and --bench-output
    #[arg(long)]
    pub bench_preset: Option<String>,

    /// Dataset directory containing WAV files for benchmark
    #[arg(long)]
    pub bench_dataset: Option<String>,

    /// Output directory for benchmark JSON reports
    #[arg(long)]
    pub bench_output: Option<String>,
}
