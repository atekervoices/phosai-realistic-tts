use anyhow::{Context, Result};
use reqwest::Client;
use serde::Serialize;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, RwLock,
};

use crate::audio;
use crate::config::{ExecutionDevice, RuntimeManifest, TtsEngineConfig};
use crate::onnx_engine::{Audio8OnnxEngine, ProgressiveAudio8Streamer};
use crate::tokenizer::Audio8PromptTokenizer;
use crate::voices::{VoiceProfile, VoiceRegistry};

pub type Audio8InferenceEngine = TtsEngine;

#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_new_tokens: usize,
    pub seed: u64,
    pub speaker_tag: Option<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 30,
            max_new_tokens: 2048,
            seed: 42,
            speaker_tag: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct SglangTtsRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    references: Option<Vec<SglangReferenceInput<'a>>>,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: u64,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct SglangReferenceInput<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vq_codes: Option<&'a Vec<Vec<i64>>>,
}

/// Thread-safe multi-worker Audio8 TTS engine supporting:
/// 1. In-process ONNX Runtime (`ort`) worker pool (`--workers N`) for GPU (`CUDA`/`TensorRT`/`DirectML`) or CPU.
/// 2. Optional `sglang-audio8` (`Audio8_TTS/sglang_omni`) continuous-batching GPU backend (`--sglang-url`).
pub struct TtsEngine {
    pub config: TtsEngineConfig,
    pub manifest: RuntimeManifest,
    pub tokenizer: Arc<Audio8PromptTokenizer>,
    pub voices: Arc<RwLock<VoiceRegistry>>,
    workers: Vec<Arc<Mutex<Audio8OnnxEngine>>>,
    next_worker: AtomicUsize,
    http_client: Client,
    pub active_provider: String,
}

impl TtsEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn from_cli(
        model_dir: &Path,
        base_dir: &Path,
        voices_dir: &Path,
        precision: Option<&str>,
        codec_precision: Option<&str>,
        device: ExecutionDevice,
        threads: usize,
        num_workers: usize,
        sglang_url: Option<String>,
        output_sample_rate: u32,
    ) -> Result<Self> {
        let cfg = TtsEngineConfig {
            model_dir: model_dir.to_path_buf(),
            voices_dir: voices_dir.to_path_buf(),
            speaker_registry_path: base_dir.join("speaker_registry.json"),
            speaker_references_root: base_dir.to_path_buf(),
            precision: precision.map(|s| s.to_string()),
            codec_precision: codec_precision.map(|s| s.to_string()),
            device,
            threads,
            num_workers,
            sglang_url,
            output_sample_rate,
            default_voice: Some("crestai_vo_tts:lug_female_10".to_string()),
        };
        Self::new(cfg)
    }

    pub fn new(config: TtsEngineConfig) -> Result<Self> {
        let manifest = RuntimeManifest::load_or_default(&config.model_dir)?;
        let tokenizer = Arc::new(Audio8PromptTokenizer::from_model_dir(
            &config.model_dir,
            &manifest,
        )?);

        let voices = Arc::new(RwLock::new(VoiceRegistry::load(
            &config.voices_dir,
            &config.speaker_registry_path,
            &config.speaker_references_root,
        )?));

        let num_workers = config.num_workers.max(1);
        let mut workers = Vec::with_capacity(num_workers);
        let mut active_provider = "CPUExecutionProvider".to_string();

        for idx in 0..num_workers {
            println!(
                "Initializing ONNX Runtime worker {}/{}...",
                idx + 1,
                num_workers
            );
            let engine = Audio8OnnxEngine::load(
                &config.model_dir,
                config.precision.as_deref(),
                config.codec_precision.as_deref(),
                config.device,
                config.threads,
            )?;
            if idx == 0 {
                active_provider = engine.active_provider.clone();
            }
            workers.push(Arc::new(Mutex::new(engine)));
        }

        if let Some(ref def_voice) = config.default_voice {
            if let Ok(mut reg) = voices.write() {
                if let Ok(mut worker0) = workers[0].lock() {
                    match reg.ensure_encoded_profile(def_voice, &mut worker0) {
                        Ok(prof) => {
                            println!(
                                "Default voice '{}' ready (language='{}').",
                                prof.name, prof.language
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "Note: Default voice '{}' not pre-encoded at startup: {}",
                                def_voice, e
                            );
                        }
                    }
                }
            }
        }

        Ok(Self {
            config,
            manifest,
            tokenizer,
            voices,
            workers,
            next_worker: AtomicUsize::new(0),
            http_client: Client::new(),
            active_provider,
        })
    }

    /// Acquire the next available ONNX Runtime worker (prefers unlocked workers, falls back to round-robin).
    fn acquire_worker(&self) -> Arc<Mutex<Audio8OnnxEngine>> {
        let len = self.workers.len();
        let start = self.next_worker.fetch_add(1, Ordering::Relaxed) % len;
        for offset in 0..len {
            let idx = (start + offset) % len;
            if self.workers[idx].try_lock().is_ok() {
                return self.workers[idx].clone();
            }
        }
        self.workers[start].clone()
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.config.output_sample_rate
    }

    pub fn native_sample_rate(&self) -> u32 {
        self.manifest.sample_rate
    }

    pub fn list_voices(&self) -> Vec<String> {
        self.voices
            .read()
            .map(|r| r.list_voices())
            .unwrap_or_default()
    }

    pub fn delete_voice(&self, name: &str) -> Result<()> {
        let mut reg = self
            .voices
            .write()
            .map_err(|_| anyhow::anyhow!("Voice registry lock poisoned"))?;
        reg.delete_voice(name)
    }

    pub fn clone_voice(
        &self,
        name: String,
        reference_text: String,
        language: String,
        wav_bytes: &[u8],
    ) -> Result<(VoiceProfile, bool)> {
        let worker = self.acquire_worker();
        let mut engine = worker
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX worker lock poisoned"))?;
        let mut reg = self
            .voices
            .write()
            .map_err(|_| anyhow::anyhow!("Voice registry lock poisoned"))?;
        reg.register_cloned_voice(name, reference_text, language, wav_bytes, &mut engine)
    }

    pub fn resolve_voice_profile(&self, voice_name: Option<&str>) -> Result<Option<VoiceProfile>> {
        let requested = voice_name
            .filter(|s| !s.trim().is_empty())
            .or(self.config.default_voice.as_deref());

        let Some(name) = requested else {
            return Ok(None);
        };

        if let Ok(reg) = self.voices.read() {
            if let Some(prof) = reg.get_profile(name) {
                if prof.codes_array(self.manifest.num_codebooks).is_some() {
                    return Ok(Some(prof.clone()));
                }
            }
        }

        let worker = self.acquire_worker();
        let mut engine = worker
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX worker lock poisoned"))?;
        let mut reg = self
            .voices
            .write()
            .map_err(|_| anyhow::anyhow!("Voice registry lock poisoned"))?;
        let prof = reg.ensure_encoded_profile(name, &mut engine)?;
        Ok(Some(prof))
    }

    /// Synthesize full audio (`f32` samples at `output_sample_rate`) for a prompt.
    pub async fn synthesize(
        &self,
        text: &str,
        voice_name: Option<&str>,
        params: GenerationParams,
    ) -> Result<Vec<f32>> {
        if let Some(ref sglang_url) = self.config.sglang_url {
            return self
                .synthesize_via_sglang(sglang_url, text, voice_name, &params)
                .await;
        }

        let mut all_samples = Vec::new();
        self.synthesize_streaming(text, voice_name, params, None, |chunk| {
            all_samples.extend_from_slice(&chunk);
            Ok(true)
        })?;
        Ok(all_samples)
    }

    /// Stream synthesized audio chunks (`f32` samples at `output_sample_rate`) sentence-by-sentence
    /// and window-by-window via SGLang continuous batching GPU backend (Step 2 architecture).
    pub fn synthesize_streaming<F>(
        &self,
        text: &str,
        voice_name: Option<&str>,
        params: GenerationParams,
        abort_flag: Option<Arc<AtomicBool>>,
        mut on_audio_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<f32>) -> Result<bool>,
    {
        // Step 2 Architecture: Stream directly from SGLang-Omni continuous batching GPU backend
        if let Some(ref sglang_url) = self.config.sglang_url {
            return self.synthesize_streaming_via_sglang(
                sglang_url,
                text,
                voice_name,
                &params,
                abort_flag,
                on_audio_chunk,
            );
        }

        let profile = self.resolve_voice_profile(voice_name)?;
        let lang_hint = profile
            .as_ref()
            .map(|p| p.name.as_str())
            .or(voice_name);
        let sentences = self.tokenizer.normalize_and_split(text, lang_hint);
        if sentences.is_empty() {
            return Ok(());
        }

        let ref_codes = profile
            .as_ref()
            .and_then(|p| p.codes_array(self.manifest.num_codebooks));
        let ref_text = profile
            .as_ref()
            .map(|p| p.reference_text.as_str())
            .filter(|s| !s.is_empty());

        let speaker_tag = params.speaker_tag.clone().or_else(|| {
            profile.as_ref().map(|p| {
                p.name
                    .split(':')
                    .nth(1)
                    .unwrap_or(&p.name)
                    .to_string()
            })
        });

        let native_sr = self.manifest.sample_rate;
        let out_sr = self.config.output_sample_rate;

        let worker = self.acquire_worker();
        let mut engine = worker
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX worker lock poisoned"))?;

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            if let Some(ref flag) = abort_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let prompt = self.tokenizer.build_prompt(
                sentence,
                ref_codes.as_ref(),
                ref_text,
                speaker_tag.as_deref(),
                &self.manifest,
            )?;

            let mut streamer = ProgressiveAudio8Streamer::new(&self.manifest);
            let mut client_open = true;

            engine.generate_frames_with_callback(
                &prompt,
                params.max_new_tokens,
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed.wrapping_add(sent_idx as u64),
                abort_flag.as_ref(),
                |frame, eng| {
                    if let Some(chunk_44k) = streamer.push_frame(frame, eng)? {
                        let out_chunk = if out_sr != native_sr {
                            audio::resample_lanczos(&chunk_44k, native_sr, out_sr)
                        } else {
                            chunk_44k
                        };
                        if !out_chunk.is_empty() {
                            client_open = on_audio_chunk(out_chunk)?;
                        }
                    }
                    Ok(client_open)
                },
            )?;

            if !client_open {
                break;
            }

            if let Some(tail_44k) = streamer.finalize(&mut engine)? {
                let out_tail = if out_sr != native_sr {
                    audio::resample_lanczos(&tail_44k, native_sr, out_sr)
                } else {
                    tail_44k
                };
                if !out_tail.is_empty() && !on_audio_chunk(out_tail)? {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Synthesize streaming audio from an ad-hoc reference WAV file or raw WAV bytes (matching `spark-tts-rs`
    /// `/v1/audio/speech/clone/upload` and `/v1/audio/speech/clone/ws`).
    pub fn synthesize_cloned_bytes_streaming<F>(
        &self,
        text: &str,
        wav_bytes: &[u8],
        reference_text: Option<&str>,
        language: Option<&str>,
        params: GenerationParams,
        abort_flag: Option<Arc<AtomicBool>>,
        mut on_audio_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<f32>) -> Result<bool>,
    {
        // Step 2 Architecture: Stream directly from SGLang-Omni backend
        if let Some(ref sglang_url) = self.config.sglang_url {
            return self.synthesize_cloned_bytes_via_sglang(
                sglang_url,
                text,
                wav_bytes,
                reference_text,
                language,
                &params,
                abort_flag,
                on_audio_chunk,
            );
        }

        let sentences = self.tokenizer.normalize_and_split(text, language);
        if sentences.is_empty() {
            return Ok(());
        }

        let native_sr = self.manifest.sample_rate;
        let out_sr = self.config.output_sample_rate;
        let samples_44k = audio::decode_wav_bytes_resampled(wav_bytes, native_sr)?;

        let worker = self.acquire_worker();
        let mut engine = worker
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX worker lock poisoned"))?;

        let ref_codes = engine.encode_reference_audio(&samples_44k)?;

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            if let Some(ref flag) = abort_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let prompt = self.tokenizer.build_prompt(
                sentence,
                Some(&ref_codes),
                reference_text,
                params.speaker_tag.as_deref(),
                &self.manifest,
            )?;

            let mut streamer = ProgressiveAudio8Streamer::new(&self.manifest);
            let mut client_open = true;

            engine.generate_frames_with_callback(
                &prompt,
                params.max_new_tokens,
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed.wrapping_add(sent_idx as u64),
                abort_flag.as_ref(),
                |frame, eng| {
                    if let Some(chunk_44k) = streamer.push_frame(frame, eng)? {
                        let out_chunk = if out_sr != native_sr {
                            audio::resample_lanczos(&chunk_44k, native_sr, out_sr)
                        } else {
                            chunk_44k
                        };
                        if !out_chunk.is_empty() {
                            client_open = on_audio_chunk(out_chunk)?;
                        }
                    }
                    Ok(client_open)
                },
            )?;

            if !client_open {
                break;
            }

            if let Some(tail_44k) = streamer.finalize(&mut engine)? {
                let out_tail = if out_sr != native_sr {
                    audio::resample_lanczos(&tail_44k, native_sr, out_sr)
                } else {
                    tail_44k
                };
                if !out_tail.is_empty() && !on_audio_chunk(out_tail)? {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Synthesize streaming audio from a reference audio path on disk (`CloneWsRequest` in `spark-tts-rs`).
    pub fn synthesize_cloned_path_streaming<F>(
        &self,
        text: &str,
        reference_audio_path: &Path,
        reference_text: Option<&str>,
        language: Option<&str>,
        params: GenerationParams,
        abort_flag: Option<Arc<AtomicBool>>,
        on_audio_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<f32>) -> Result<bool>,
    {
        let wav_bytes = std::fs::read(reference_audio_path).with_context(|| {
            format!(
                "Failed to read reference audio file at {}",
                reference_audio_path.display()
            )
        })?;
        self.synthesize_cloned_bytes_streaming(
            text,
            &wav_bytes,
            reference_text,
            language,
            params,
            abort_flag,
            on_audio_chunk,
        )
    }

    async fn synthesize_via_sglang(
        &self,
        base_url: &str,
        text: &str,
        voice_name: Option<&str>,
        params: &GenerationParams,
    ) -> Result<Vec<f32>> {
        let profile = self.resolve_voice_profile(voice_name)?;
        let lang_hint = profile
            .as_ref()
            .map(|p| p.name.as_str())
            .or(voice_name);
        let sentences = self.tokenizer.normalize_and_split(text, lang_hint);
        let normalized_text = sentences.join(" ");

        let references = profile.as_ref().and_then(|prof| {
            prof.vq_codes.as_ref().map(|codes| {
                vec![SglangReferenceInput {
                    text: if prof.reference_text.is_empty() {
                        None
                    } else {
                        Some(prof.reference_text.as_str())
                    },
                    vq_codes: Some(codes),
                }]
            })
        });

        let req_body = SglangTtsRequest {
            text: &normalized_text,
            references,
            max_new_tokens: params.max_new_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            seed: params.seed,
            stream: false,
        };

        let url = format!("{}/v1/audio/speech", base_url.trim_end_matches('/'));
        let resp = self
            .http_client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .with_context(|| format!("Failed to reach sglang-audio8 at {}", url))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_txt = resp.text().await.unwrap_or_default();
            anyhow::bail!("sglang-audio8 returned {}: {}", status, err_txt);
        }

        let wav_bytes = resp.bytes().await?;
        audio::decode_wav_bytes_resampled(&wav_bytes, self.config.output_sample_rate)
    }

    fn synthesize_streaming_via_sglang<F>(
        &self,
        sglang_url: &str,
        text: &str,
        voice_name: Option<&str>,
        params: &GenerationParams,
        abort_flag: Option<Arc<AtomicBool>>,
        mut on_audio_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<f32>) -> Result<bool>,
    {
        let profile = self.resolve_voice_profile(voice_name)?;
        let lang_hint = profile.as_ref().map(|p| p.name.as_str()).or(voice_name);
        let sentences = self.tokenizer.normalize_and_split(text, lang_hint);
        if sentences.is_empty() {
            return Ok(());
        }

        let out_sr = self.config.output_sample_rate;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        for sentence in &sentences {
            if let Some(ref flag) = abort_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let mut references = Vec::new();
            if let Some(ref prof) = profile {
                let mut ref_map = serde_json::Map::new();
                if !prof.reference_text.is_empty() {
                    ref_map.insert("text".to_string(), serde_json::Value::String(prof.reference_text.clone()));
                }
                if let Some(ref codes) = prof.vq_codes {
                    ref_map.insert("vq_codes".to_string(), serde_json::to_value(codes)?);
                } else if let Some(ref path) = prof.audio_path {
                    ref_map.insert("audio_path".to_string(), serde_json::Value::String(path.to_string_lossy().to_string()));
                }
                if !ref_map.is_empty() {
                    references.push(serde_json::Value::Object(ref_map));
                }
            }

            let mut body_map = serde_json::Map::new();
            body_map.insert("model".to_string(), serde_json::Value::String("audio8/tts-0.6b".to_string()));
            body_map.insert("input".to_string(), serde_json::Value::String(sentence.clone()));
            body_map.insert("text".to_string(), serde_json::Value::String(sentence.clone()));
            body_map.insert("response_format".to_string(), serde_json::Value::String("pcm".to_string()));
            body_map.insert("stream".to_string(), serde_json::Value::Bool(true));
            body_map.insert("max_new_tokens".to_string(), serde_json::json!(params.max_new_tokens));
            body_map.insert("temperature".to_string(), serde_json::json!(params.temperature));
            body_map.insert("top_p".to_string(), serde_json::json!(params.top_p));
            body_map.insert("top_k".to_string(), serde_json::json!(params.top_k));
            body_map.insert("seed".to_string(), serde_json::json!(params.seed));

            if !references.is_empty() {
                body_map.insert("references".to_string(), serde_json::Value::Array(references));
            }

            let url = format!("{}/v1/audio/speech", sglang_url.trim_end_matches('/'));
            let resp = client
                .post(&url)
                .json(&serde_json::Value::Object(body_map))
                .send()
                .with_context(|| format!("Failed to connect to SGLang backend at {}", url))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let txt = resp.text().unwrap_or_default();
                anyhow::bail!("SGLang backend returned status {}: {}", status, txt);
            }

            use std::io::BufRead;
            let mut reader = std::io::BufReader::new(resp);
            let mut line = String::new();

            loop {
                if let Some(ref flag) = abort_flag {
                    if flag.load(Ordering::Relaxed) {
                        break;
                    }
                }

                line.clear();
                let bytes_read = reader.read_line(&mut line)?;
                if bytes_read == 0 {
                    break;
                }

                let trimmed = line.trim();
                if !trimmed.starts_with("data: ") {
                    continue;
                }
                let data = &trimmed[6..];
                if data == "[DONE]" {
                    break;
                }

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(audio_obj) = val.get("audio") {
                        let sample_rate = audio_obj
                            .get("sample_rate")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(44100) as u32;
                        if let Some(b64_str) = audio_obj.get("data").and_then(|v| v.as_str()) {
                            use base64::Engine;
                            if let Ok(pcm_bytes) = base64::engine::general_purpose::STANDARD.decode(b64_str) {
                                let samples_native = audio::pcm16le_bytes_to_samples(&pcm_bytes);
                                let out_samples = if out_sr != sample_rate {
                                    audio::resample_lanczos(&samples_native, sample_rate, out_sr)
                                } else {
                                    samples_native
                                };
                                if !out_samples.is_empty() {
                                    let keep_going = on_audio_chunk(out_samples)?;
                                    if !keep_going {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn synthesize_cloned_bytes_via_sglang<F>(
        &self,
        sglang_url: &str,
        text: &str,
        wav_bytes: &[u8],
        reference_text: Option<&str>,
        language: Option<&str>,
        params: &GenerationParams,
        abort_flag: Option<Arc<AtomicBool>>,
        mut on_audio_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<f32>) -> Result<bool>,
    {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(wav_bytes);
        let data_uri = format!("data:audio/wav;base64,{}", b64);

        let sentences = self.tokenizer.normalize_and_split(text, language);
        if sentences.is_empty() {
            return Ok(());
        }

        let out_sr = self.config.output_sample_rate;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        for sentence in &sentences {
            if let Some(ref flag) = abort_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let mut ref_map = serde_json::Map::new();
            ref_map.insert("audio_path".to_string(), serde_json::Value::String(data_uri.clone()));
            if let Some(ref txt) = reference_text {
                ref_map.insert("text".to_string(), serde_json::Value::String(txt.to_string()));
            }
            let references = vec![serde_json::Value::Object(ref_map)];

            let mut body_map = serde_json::Map::new();
            body_map.insert("model".to_string(), serde_json::Value::String("audio8/tts-0.6b".to_string()));
            body_map.insert("input".to_string(), serde_json::Value::String(sentence.clone()));
            body_map.insert("text".to_string(), serde_json::Value::String(sentence.clone()));
            body_map.insert("response_format".to_string(), serde_json::Value::String("pcm".to_string()));
            body_map.insert("stream".to_string(), serde_json::Value::Bool(true));
            body_map.insert("max_new_tokens".to_string(), serde_json::json!(params.max_new_tokens));
            body_map.insert("temperature".to_string(), serde_json::json!(params.temperature));
            body_map.insert("top_p".to_string(), serde_json::json!(params.top_p));
            body_map.insert("top_k".to_string(), serde_json::json!(params.top_k));
            body_map.insert("seed".to_string(), serde_json::json!(params.seed));
            body_map.insert("references".to_string(), serde_json::Value::Array(references));

            let url = format!("{}/v1/audio/speech", sglang_url.trim_end_matches('/'));
            let resp = client
                .post(&url)
                .json(&serde_json::Value::Object(body_map))
                .send()
                .with_context(|| format!("Failed to connect to SGLang backend at {}", url))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let txt = resp.text().unwrap_or_default();
                anyhow::bail!("SGLang backend returned status {}: {}", status, txt);
            }

            use std::io::BufRead;
            let mut reader = std::io::BufReader::new(resp);
            let mut line = String::new();

            loop {
                if let Some(ref flag) = abort_flag {
                    if flag.load(Ordering::Relaxed) {
                        break;
                    }
                }

                line.clear();
                let bytes_read = reader.read_line(&mut line)?;
                if bytes_read == 0 {
                    break;
                }

                let trimmed = line.trim();
                if !trimmed.starts_with("data: ") {
                    continue;
                }
                let data = &trimmed[6..];
                if data == "[DONE]" {
                    break;
                }

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(audio_obj) = val.get("audio") {
                        let sample_rate = audio_obj
                            .get("sample_rate")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(44100) as u32;
                        if let Some(b64_str) = audio_obj.get("data").and_then(|v| v.as_str()) {
                            use base64::Engine;
                            if let Ok(pcm_bytes) = base64::engine::general_purpose::STANDARD.decode(b64_str) {
                                let samples_native = audio::pcm16le_bytes_to_samples(&pcm_bytes);
                                let out_samples = if out_sr != sample_rate {
                                    audio::resample_lanczos(&samples_native, sample_rate, out_sr)
                                } else {
                                    samples_native
                                };
                                if !out_samples.is_empty() {
                                    let keep_going = on_audio_chunk(out_samples)?;
                                    if !keep_going {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

}
