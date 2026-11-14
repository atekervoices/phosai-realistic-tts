use anyhow::Result;
use clap::Parser;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use audio8_tts_rs::config::ExecutionDevice;
use audio8_tts_rs::inference::Audio8InferenceEngine;
use audio8_tts_rs::server::run_server;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Audio8-TTS-RS: Native Rust + LibTorch (tch-rs) CUDA/CPU Streaming TTS Server"
)]
struct Args {
    /// Directory containing exported TorchScript models (`runtime_manifest.json`, `slow_ar_*.pt`, `fast_ar_*.pt`, `codec_decoder_*.pt`, `tokenizer.json`)
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Base directory containing `speaker_registry.json` and `speaker_references/*.wav`
    #[arg(long)]
    base_dir: Option<PathBuf>,

    /// Directory containing pre-encoded voice profiles
    #[arg(long)]
    voices_dir: Option<PathBuf>,

    /// Execution device for LibTorch (`auto`, `gpu`/`cuda`, or `cpu`)
    #[arg(long)]
    device: Option<String>,

    /// DualAR TorchScript precision (`fp16` or `fp32`)
    #[arg(long)]
    precision: Option<String>,

    /// Codec decoder TorchScript precision (`fp16` or `fp32`)
    #[arg(long)]
    codec_precision: Option<String>,

    /// CPU intra-op thread count per worker
    #[arg(long)]
    threads: Option<usize>,

    /// Number of parallel LibTorch worker instances in the multi-concurrency pool
    #[arg(long)]
    workers: Option<usize>,

    /// Default output PCM sample rate (`16000` for drop-in LiveKit `sparktts.py` compatibility, or `44100` for native Audio8 studio quality)
    #[arg(long)]
    output_sample_rate: Option<u32>,

    /// HTTP & WebSocket server port (default `18473`)
    #[arg(short, long)]
    port: Option<u16>,
}

fn env_or<T: std::str::FromStr>(cli: Option<T>, key: &str, default: T) -> T {
    cli.or_else(|| env::var(key).ok().and_then(|v| v.parse::<T>().ok()))
        .unwrap_or(default)
}

fn env_opt(cli: Option<String>, key: &str) -> Option<String> {
    cli.or_else(|| env::var(key).ok().filter(|v| !v.trim().is_empty()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let model_dir = args
        .model_dir
        .or_else(|| env::var("ARKTTS_MODEL_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("model"));

    let base_dir = args
        .base_dir
        .or_else(|| env::var("ARKTTS_BASE_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));

    let voices_dir = args
        .voices_dir
        .or_else(|| env::var("ARKTTS_VOICES_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("voices"));

    let device_str = args
        .device
        .or_else(|| env::var("ARKTTS_DEVICE").ok())
        .unwrap_or_else(|| "auto".to_string());
    let device = ExecutionDevice::from_str_lossy(&device_str);

    let precision = env_opt(args.precision, "ARKTTS_PRECISION");
    let codec_precision = env_opt(args.codec_precision, "ARKTTS_CODEC_PRECISION");
    let threads = env_or(args.threads, "ARKTTS_THREADS", 4);
    let workers = env_or(args.workers, "ARKTTS_WORKERS", 1);
    let output_sample_rate = env_or(args.output_sample_rate, "OUTPUT_SAMPLE_RATE", 16000);
    let port = env_or(args.port, "PORT", 18473);

    println!("============================================================");
    println!("Initializing Audio8-TTS-RS Native LibTorch Inference Engine");
    println!("  Model Dir          : {}", model_dir.display());
    println!("  Voices Dir         : {}", voices_dir.display());
    println!("  Device Preference  : {:?}", device);
    println!("  LibTorch Workers   : {}", workers);
    println!("  Output Sample Rate : {} Hz", output_sample_rate);
    println!("============================================================");

    let engine = Arc::new(Audio8InferenceEngine::from_cli(
        &model_dir,
        &base_dir,
        &voices_dir,
        precision.as_deref(),
        codec_precision.as_deref(),
        device,
        threads,
        workers,
        output_sample_rate,
    )?);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    run_server(engine, addr).await;
    Ok(())
}
