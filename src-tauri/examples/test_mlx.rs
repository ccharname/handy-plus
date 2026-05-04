//! Standalone MLX audio bridge test — bypasses Tauri/.app/hardened-runtime.
//!
//! Usage:
//!   cargo run --release --example test_mlx -- /path/to/16khz-mono.wav voxtral-mini-4b-4bit
//!
//! Dev binary is adhoc-signed only; no hardened runtime, no kernel SIGKILL on
//! JIT page allocation. If this segfaults / aborts here, it's a real Swift
//! bridge bug, not an entitlement issue.

use handy_app_lib::mlx_audio;
use std::path::PathBuf;

fn main() {
    eprintln!("=== test_mlx — direct mlx_audio bridge probe ===");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: test_mlx <wav_path> <model_id>");
        eprintln!("  model_id ∈ {{voxtral-mini-4b-4bit, qwen3-asr-06b-8bit}}");
        std::process::exit(2);
    }
    let wav_path = PathBuf::from(&args[1]);
    let model_id = &args[2];

    eprintln!("Step 1: bridge_version()");
    match mlx_audio::bridge_version() {
        Ok(v) => eprintln!("  PASS: {}", v),
        Err(e) => {
            eprintln!("  FAIL: {}", e);
            std::process::exit(1);
        }
    }

    eprintln!("Step 2: transcribe_file({:?}, {})", wav_path, model_id);
    let t0 = std::time::Instant::now();
    match mlx_audio::transcribe_file(&wav_path, model_id) {
        Ok(text) => {
            let dt = t0.elapsed();
            eprintln!("  PASS in {:?}", dt);
            eprintln!("  TEXT: {}", text);
        }
        Err(e) => {
            let dt = t0.elapsed();
            eprintln!("  FAIL in {:?}: {}", dt, e);
            std::process::exit(1);
        }
    }
}
