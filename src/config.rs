use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Execution device preference for LibTorch (`tch-rs`) inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionDevice {
    /// Use CUDA GPU automatically when available, otherwise fall back to CPU.
    Auto,
    /// Require or prioritize CUDA GPU execution (`Device::Cuda(0)`).
    Gpu,
    /// Run strictly on CPU (`Device::Cpu`).
    Cpu,
}

impl ExecutionDevice {
    pub fn from_str_lossy(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "gpu" | "cuda" => Self::Gpu,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

/// Manifest describing exported Audio8 TorchScript runtime files and model constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeManifest {
    #[serde(default = "default_model_id")]
    pub model_id: String,
    #[serde(default = "default_model_fingerprint")]
    pub model_fingerprint: String,
    #[serde(default = "default_precision")]
    pub default_precision: String,
    #[serde(default = "default_available_precisions")]
    pub available_precisions: Vec<String>,
    #[serde(default = "default_codec_precision")]
    pub default_codec_precision: String,
    #[serde(default = "default_available_codec_precisions")]
    pub available_codec_precisions: Vec<String>,

    #[serde(default)]
    pub slow_decode_models: HashMap<String, String>,
    #[serde(default)]
    pub slow_decode_model: Option<String>,
    #[serde(default)]
    pub fast_models: HashMap<String, String>,
    #[serde(default)]
    pub fast_model: Option<String>,
    #[serde(default)]
    pub codec_models: HashMap<String, String>,
    #[serde(default)]
    pub codec_encoder_model: Option<String>,

    /// Native Audio8 codec sample rate (44100 Hz).
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// Number of VQ codebooks (10 for Audio8-TTS-Realistic-Air).
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,
    /// Entries per codebook (4096).
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,
    /// Samples per codec frame (2048 at 44.1kHz -> ~46.44ms per frame).
    #[serde(default = "default_codec_hop_length")]
    pub codec_hop_length: usize,
    /// Sliding context window in frames for streaming vocoder decoding (48).
    #[serde(default = "default_stream_context_frames")]
    pub stream_context_frames: usize,
    /// Guard frames trimmed from the right edge of streaming decode windows (1).
    #[serde(default = "default_stream_guard_frames")]
    pub stream_guard_frames: usize,

    /// Slow AR transformer layer count (24 for 0.6B Realistic-Air).
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    /// Fast AR codebook transformer layer count (4).
    #[serde(default = "default_num_fast_layers")]
    pub num_fast_layers: usize,
    /// Number of KV heads in Slow AR (2).
    #[serde(default = "default_n_local_heads")]
    pub n_local_heads: usize,
    /// Head dimension in Slow AR (64).
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    /// Head dimension in Fast AR (64).
    #[serde(default = "default_fast_head_dim")]
    pub fast_head_dim: usize,
    /// Hidden size of Slow AR (896 for 0.6B Realistic-Air).
    #[serde(default = "default_dim")]
    pub dim: usize,
    /// Maximum sequence length for Slow AR KV cache (2048).
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// First token ID of `<|semantic:0|>`.
    #[serde(default = "default_semantic_begin_id")]
    pub semantic_begin_id: i64,
    /// Last token ID of `<|semantic:4095|>`.
    #[serde(default = "default_semantic_end_id")]
    pub semantic_end_id: i64,
    /// Token ID of `<|im_end|>` (EOS).
    #[serde(default = "default_im_end_id")]
    pub im_end_id: i64,

    /// Layout of Slow AR output logits:
    /// - `"relative_semantic_then_eos"`: `[0..4096]` = semantic tokens, `[4096]` = `<|im_end|>`
    /// - `"eos_then_relative_semantic"`: `[0]` = `<|im_end|>`, `[1..4097]` = semantic tokens
    /// - `"full_vocab"`: full vocabulary logits where `semantic_begin_id..=semantic_end_id` and `im_end_id` are indexed directly
    #[serde(default = "default_slow_logits_layout")]
    pub slow_logits_layout: String,

    /// Repetition Aware Sampling (RAS) window size (default 10).
    #[serde(default = "default_ras_window_size")]
    pub ras_window_size: usize,
    #[serde(default = "default_ras_temperature")]
    pub ras_temperature: f32,
    #[serde(default = "default_ras_top_p")]
    pub ras_top_p: f32,
}

fn default_model_id() -> String {
    "phosai/realistic-air-v1".to_string()
}
fn default_model_fingerprint() -> String {
    "phosai-realistic-air-v1-0.6b".to_string()
}
fn default_precision() -> String {
    "fp16".to_string()
}
fn default_available_precisions() -> Vec<String> {
    vec!["fp16".to_string(), "fp32".to_string()]
}
fn default_codec_precision() -> String {
    "fp16".to_string()
}
fn default_available_codec_precisions() -> Vec<String> {
    vec!["fp16".to_string(), "fp32".to_string()]
}
fn default_sample_rate() -> u32 {
    44100
}
fn default_num_codebooks() -> usize {
    10
}
fn default_codebook_size() -> usize {
    4096
}
fn default_codec_hop_length() -> usize {
    2048
}
fn default_stream_context_frames() -> usize {
    48
}
fn default_stream_guard_frames() -> usize {
    1
}
fn default_num_layers() -> usize {
    24
}
fn default_num_fast_layers() -> usize {
    4
}
fn default_n_local_heads() -> usize {
    2
}
fn default_head_dim() -> usize {
    64
}
fn default_fast_head_dim() -> usize {
    64
}
fn default_dim() -> usize {
    896
}
fn default_max_seq_len() -> usize {
    2048
}
fn default_semantic_begin_id() -> i64 {
    151678
}
fn default_semantic_end_id() -> i64 {
    155773
}
fn default_im_end_id() -> i64 {
    151645
}
fn default_slow_logits_layout() -> String {
    "relative_semantic_then_eos".to_string()
}
fn default_ras_window_size() -> usize {
    10
}
fn default_ras_temperature() -> f32 {
    1.0
}
fn default_ras_top_p() -> f32 {
    0.9
}

impl Default for RuntimeManifest {
    fn default() -> Self {
        Self {
            model_id: default_model_id(),
            model_fingerprint: default_model_fingerprint(),
            default_precision: default_precision(),
            available_precisions: default_available_precisions(),
            default_codec_precision: default_codec_precision(),
            available_codec_precisions: default_available_codec_precisions(),
            slow_decode_models: HashMap::new(),
            slow_decode_model: None,
            fast_models: HashMap::new(),
            fast_model: None,
            codec_models: HashMap::new(),
            codec_encoder_model: None,
            sample_rate: default_sample_rate(),
            num_codebooks: default_num_codebooks(),
            codebook_size: default_codebook_size(),
            codec_hop_length: default_codec_hop_length(),
            stream_context_frames: default_stream_context_frames(),
            stream_guard_frames: default_stream_guard_frames(),
            num_layers: default_num_layers(),
            num_fast_layers: default_num_fast_layers(),
            n_local_heads: default_n_local_heads(),
            head_dim: default_head_dim(),
            fast_head_dim: default_fast_head_dim(),
            dim: default_dim(),
            max_seq_len: default_max_seq_len(),
            semantic_begin_id: default_semantic_begin_id(),
            semantic_end_id: default_semantic_end_id(),
            im_end_id: default_im_end_id(),
            slow_logits_layout: default_slow_logits_layout(),
            ras_window_size: default_ras_window_size(),
            ras_temperature: default_ras_temperature(),
            ras_top_p: default_ras_top_p(),
        }
    }
}

impl RuntimeManifest {
    /// Load `runtime_manifest.json` from `model_dir`, or fallback to `config.json` / defaults if not present yet.
    pub fn load_or_default(model_dir: &Path) -> Result<Self> {
        let manifest_path = model_dir.join("runtime_manifest.json");
        if manifest_path.is_file() {
            let content = fs::read_to_string(&manifest_path)
                .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
            let manifest: RuntimeManifest = serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;
            return Ok(manifest);
        }

        // Fallback: check if HuggingFace `config.json` exists in `model_dir`
        let mut manifest = RuntimeManifest::default();
        let hf_config_path = model_dir.join("config.json");
        if hf_config_path.is_file() {
            if let Ok(content) = fs::read_to_string(&hf_config_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(v) = val.get("n_layer").and_then(|v| v.as_u64()) {
                        manifest.num_layers = v as usize;
                    }
                    if let Some(v) = val.get("n_fast_layer").and_then(|v| v.as_u64()) {
                        manifest.num_fast_layers = v as usize;
                    }
                    if let Some(v) = val.get("n_local_heads").and_then(|v| v.as_u64()) {
                        manifest.n_local_heads = v as usize;
                    }
                    if let Some(v) = val.get("head_dim").and_then(|v| v.as_u64()) {
                        manifest.head_dim = v as usize;
                    }
                    if let Some(v) = val.get("fast_head_dim").and_then(|v| v.as_u64()) {
                        manifest.fast_head_dim = v as usize;
                    }
                    if let Some(v) = val.get("dim").and_then(|v| v.as_u64()) {
                        manifest.dim = v as usize;
                    }
                    if let Some(v) = val.get("max_seq_len").and_then(|v| v.as_u64()) {
                        manifest.max_seq_len = v as usize;
                    }
                    if let Some(v) = val.get("num_codebooks").and_then(|v| v.as_u64()) {
                        manifest.num_codebooks = v as usize;
                    }
                    if let Some(v) = val.get("codebook_size").and_then(|v| v.as_u64()) {
                        manifest.codebook_size = v as usize;
                    }
                    if let Some(v) = val.get("semantic_begin_id").and_then(|v| v.as_i64()) {
                        manifest.semantic_begin_id = v;
                    }
                    if let Some(v) = val.get("semantic_end_id").and_then(|v| v.as_i64()) {
                        manifest.semantic_end_id = v;
                    }
                    if let Some(v) = val.get("eos_token_id").and_then(|v| v.as_i64()) {
                        manifest.im_end_id = v;
                    }
                }
            }
        }
        Ok(manifest)
    }

    pub fn resolve_slow_model(&self, model_dir: &Path, precision: &str) -> PathBuf {
        if let Some(name) = self.slow_decode_models.get(precision) {
            let candidate = model_dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        if let Some(ref name) = self.slow_decode_model {
            let candidate = model_dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        let by_precision = model_dir.join(format!("slow_ar_{precision}.pt"));
        if by_precision.is_file() {
            return by_precision;
        }
        let fp16 = model_dir.join("slow_ar_fp16.pt");
        if fp16.is_file() {
            return fp16;
        }
        let fp32 = model_dir.join("slow_ar_fp32.pt");
        if fp32.is_file() {
            return fp32;
        }
        let default_name = model_dir.join("slow_ar.pt");
        if default_name.is_file() {
            return default_name;
        }
        by_precision
    }

    pub fn resolve_fast_model(&self, model_dir: &Path, precision: &str) -> PathBuf {
        if let Some(name) = self.fast_models.get(precision) {
            let candidate = model_dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        if let Some(ref name) = self.fast_model {
            let candidate = model_dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        let by_precision = model_dir.join(format!("fast_ar_{precision}.pt"));
        if by_precision.is_file() {
            return by_precision;
        }
        let fp16 = model_dir.join("fast_ar_fp16.pt");
        if fp16.is_file() {
            return fp16;
        }
        let fp32 = model_dir.join("fast_ar_fp32.pt");
        if fp32.is_file() {
            return fp32;
        }
        let default_name = model_dir.join("fast_ar.pt");
        if default_name.is_file() {
            return default_name;
        }
        by_precision
    }

    pub fn resolve_codec_decoder_model(&self, model_dir: &Path, codec_precision: &str) -> PathBuf {
        if let Some(name) = self.codec_models.get(codec_precision) {
            let candidate = model_dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        let by_precision = model_dir.join(format!("codec_decoder_{codec_precision}.pt"));
        if by_precision.is_file() {
            return by_precision;
        }
        let fp16 = model_dir.join("codec_decoder_fp16.pt");
        if fp16.is_file() {
            return fp16;
        }
        let fp32 = model_dir.join("codec_decoder_fp32.pt");
        if fp32.is_file() {
            return fp32;
        }
        model_dir.join("codec_decoder.pt")
    }

    pub fn resolve_codec_encoder_model(&self, model_dir: &Path) -> Option<PathBuf> {
        if let Some(ref name) = self.codec_encoder_model {
            let candidate = model_dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        let candidates = [
            model_dir.join("registration").join("codec_encoder_fp32.pt"),
            model_dir.join("registration").join("codec_encoder.pt"),
            model_dir.join("registration").join("codec_encoder_fp16.pt"),
            model_dir.join("codec_encoder_fp32.pt"),
            model_dir.join("codec_encoder.pt"),
            model_dir.join("codec_encoder_fp16.pt"),
        ];
        candidates.into_iter().find(|p| p.is_file())
    }
}

/// Top-level runtime configuration for `TtsEngine`.
#[derive(Debug, Clone)]
pub struct TtsEngineConfig {
    pub model_dir: PathBuf,
    pub voices_dir: PathBuf,
    pub speaker_registry_path: PathBuf,
    pub speaker_references_root: PathBuf,
    pub precision: Option<String>,
    pub codec_precision: Option<String>,
    pub device: ExecutionDevice,
    pub threads: usize,
    pub num_workers: usize,
    pub output_sample_rate: u32,
    pub default_voice: Option<String>,
}

impl Default for TtsEngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("model"),
            voices_dir: PathBuf::from("voices"),
            speaker_registry_path: PathBuf::from("speaker_registry.json"),
            speaker_references_root: PathBuf::from("."),
            precision: None,
            codec_precision: None,
            device: ExecutionDevice::Auto,
            threads: 4,
            num_workers: 1,
            output_sample_rate: 16000,
            default_voice: Some("crestai_vo_tts:lug_female_10".to_string()),
        }
    }
}

