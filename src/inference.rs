use anyhow::{Context, Result};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, RwLock,
};

use crate::audio;
use crate::config::{ExecutionDevice, RuntimeManifest, TtsEngineConfig};
use crate::tch_engine::{Audio8TchEngine, ProgressiveAudio8Streamer};
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

/// Thread-safe multi-worker Audio8 TTS engine backed natively by LibTorch (`tch-rs`)
/// running SlowAR + fused FastARLoop + sliding-window streaming ArkttsCodec on CUDA or CPU.
pub struct TtsEngine {
    pub config: TtsEngineConfig,
    pub manifest: RuntimeManifest,
    pub tokenizer: Arc<Audio8PromptTokenizer>,
    pub voices: Arc<RwLock<VoiceRegistry>>,
    workers: Vec<Arc<Mutex<Audio8TchEngine>>>,
    next_worker: AtomicUsize,
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
        let mut active_provider = "LibTorchCPU".to_string();

        for idx in 0..num_workers {
            println!(
                "Initializing LibTorch worker {}/{}...",
                idx + 1,
                num_workers
            );
            let engine = Audio8TchEngine::load(
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
            active_provider,
        })
    }

    /// Acquire the next available LibTorch worker (prefers unlocked workers, falls back to round-robin).
    fn acquire_worker(&self) -> Arc<Mutex<Audio8TchEngine>> {
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
            .map_err(|_| anyhow::anyhow!("LibTorch worker lock poisoned"))?;
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
            .map_err(|_| anyhow::anyhow!("LibTorch worker lock poisoned"))?;
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
        let mut all_samples = Vec::new();
        self.synthesize_streaming(text, voice_name, params, None, |chunk| {
            all_samples.extend_from_slice(&chunk);
            Ok(true)
        })?;
        Ok(all_samples)
    }

    /// Stream synthesized audio chunks (`f32` samples at `output_sample_rate`) sentence-by-sentence
    /// and window-by-window directly from the in-process LibTorch GPU/CPU engine.
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
            .map_err(|_| anyhow::anyhow!("LibTorch worker lock poisoned"))?;

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
            let mut resampler = (out_sr != native_sr)
                .then(|| audio::StreamingLanczosResampler::new(native_sr, out_sr));
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
                        client_open = emit_resampled_chunk(
                            &mut resampler,
                            chunk_44k,
                            false,
                            &mut on_audio_chunk,
                        )?;
                    }
                    Ok(client_open)
                },
            )?;

            if !client_open {
                break;
            }

            let tail_44k = streamer.finalize(&mut engine)?.unwrap_or_default();
            if !emit_resampled_chunk(&mut resampler, tail_44k, true, &mut on_audio_chunk)? {
                break;
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
            .map_err(|_| anyhow::anyhow!("LibTorch worker lock poisoned"))?;

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
            let mut resampler = (out_sr != native_sr)
                .then(|| audio::StreamingLanczosResampler::new(native_sr, out_sr));
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
                        client_open = emit_resampled_chunk(
                            &mut resampler,
                            chunk_44k,
                            false,
                            &mut on_audio_chunk,
                        )?;
                    }
                    Ok(client_open)
                },
            )?;

            if !client_open {
                break;
            }

            let tail_44k = streamer.finalize(&mut engine)?.unwrap_or_default();
            if !emit_resampled_chunk(&mut resampler, tail_44k, true, &mut on_audio_chunk)? {
                break;
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
}

fn emit_resampled_chunk<F>(
    resampler: &mut Option<audio::StreamingLanczosResampler>,
    samples_44k: Vec<f32>,
    is_final: bool,
    on_audio_chunk: &mut F,
) -> Result<bool>
where
    F: FnMut(Vec<f32>) -> Result<bool>,
{
    let out_chunk = if let Some(r) = resampler.as_mut() {
        r.push(&samples_44k, is_final)
    } else {
        samples_44k
    };
    if out_chunk.is_empty() {
        Ok(true)
    } else {
        on_audio_chunk(out_chunk)
    }
}
