use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2, Array3, Array4};
use ort::{
    environment::Environment,
    ep::{self, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    value::{Tensor, ValueType},
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::borrow::Cow;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock,
};

use crate::audio;
use crate::config::{ExecutionDevice, RuntimeManifest};

#[cfg(target_os = "linux")]
#[link(name = "ort_compat", kind = "static")]
extern "C" {}

static ORT_ENV: OnceLock<Environment> = OnceLock::new();

pub fn get_ort_environment() -> Result<&'static Environment> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .with_name("audio8-tts-rs")
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to initialize ONNX Runtime environment: {}", e))?;
    let _ = ORT_ENV.set(env);
    Ok(ORT_ENV.get().unwrap())
}

/// Build an ONNX Runtime `Session` configured for GPU (`CUDA`, `TensorRT`, `DirectML`) and/or `CPU`.
pub fn build_ort_session(
    model_path: &Path,
    device: ExecutionDevice,
    threads: usize,
) -> Result<(Session, String)> {
    if !model_path.is_file() {
        anyhow::bail!("ONNX model file not found: {}", model_path.display());
    }
    let env = get_ort_environment()?;
    let intra = threads.max(1);
    let inter = (threads / 2).max(1);

    let mut builder = Session::builder(env)
        .map_err(|e| anyhow::anyhow!("Failed to create ORT SessionBuilder: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap_or_else(|e| e.recover())
        .with_intra_threads(intra)
        .unwrap_or_else(|e| e.recover())
        .with_inter_threads(inter)
        .unwrap_or_else(|e| e.recover());

    let mut active_provider = "CPUExecutionProvider".to_string();

    if matches!(device, ExecutionDevice::Gpu | ExecutionDevice::Auto) {
        let mut providers = Vec::new();
        let mut cuda_ok = false;

        // Try CUDA first if available
        match std::panic::catch_unwind(|| ep::CUDA::is_available()) {
            Ok(Ok(true)) => {
                match std::panic::catch_unwind(|| ep::CUDA::default().build()) {
                    Ok(ep) => {
                        providers.push(ep);
                        active_provider = "CUDAExecutionProvider".to_string();
                        cuda_ok = true;
                    }
                    Err(_) => {
                        eprintln!("[Warn] Failed to build CUDA execution provider, falling back.");
                    }
                }
            }
            _ => {}
        }

        #[cfg(target_os = "windows")]
        if !cuda_ok {
            match std::panic::catch_unwind(|| ep::DirectML::is_available()) {
                Ok(Ok(true)) => {
                    match std::panic::catch_unwind(|| ep::DirectML::default().build()) {
                        Ok(ep) => {
                            providers.push(ep);
                            active_provider = "DirectMLExecutionProvider".to_string();
                        }
                        Err(_) => {}
                    }
                }
                _ => {}
            }
        }

        // Always add CPU as final fallback provider
        providers.push(ep::CPU::default().build());

        builder = builder
            .with_execution_providers(providers)
            .unwrap_or_else(|e| {
                active_provider = "CPUExecutionProvider".to_string();
                e.recover()
            });
    } else {
        let cpu_ep = ep::CPU::default().build();
        builder = builder
            .with_execution_providers([cpu_ep])
            .unwrap_or_else(|e| e.recover());
    }

    let session = builder.commit_from_file(model_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load ONNX session from {}: {}",
            model_path.display(),
            e
        )
    })?;

    Ok((session, active_provider))
}

/// Top-K / Top-P / Temperature Gumbel-Max sampler matching `Audio8_TTS/onnx_runtime/arktts_runtime/runtime.py`.
pub fn sample_logits(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng: &mut StdRng,
) -> Result<usize> {
    if logits.is_empty() || !logits.iter().any(|v| v.is_finite()) {
        anyhow::bail!("Model returned empty or non-finite logits");
    }

    // Greedy fastpath for very low temperature
    if temperature <= 0.02 || top_k == 1 {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return Ok(best_idx);
    }

    let temp = (temperature as f64).max(1e-5);
    let p_thresh = (top_p as f64).clamp(1e-5, 1.0);
    let k_limit = top_k.clamp(1, logits.len());

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_sorted = logits[order[0]] as f64;
    let mut probs: Vec<f64> = order
        .iter()
        .take(k_limit)
        .map(|&idx| {
            let v = logits[idx] as f64;
            if v.is_finite() {
                (v - max_sorted).exp()
            } else {
                0.0
            }
        })
        .collect();

    let sum_probs: f64 = probs.iter().sum();
    if sum_probs <= 0.0 {
        return Ok(order[0]);
    }
    for p in &mut probs {
        *p /= sum_probs;
    }

    let mut cumsum = 0.0;
    let mut keep_count = probs.len();
    for (rank, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > p_thresh && rank > 0 {
            keep_count = rank;
            break;
        }
    }
    let kept_indices = &order[..keep_count.max(1)];

    let max_kept = logits[kept_indices[0]] as f64 / temp;
    let mut best_token = kept_indices[0];
    let mut best_score = f64::NEG_INFINITY;

    for &idx in kept_indices {
        let scaled = (logits[idx] as f64 / temp) - max_kept;
        let u: f64 = ((rng.next_u64() as f64) / (u64::MAX as f64)).clamp(1e-12, 1.0 - 1e-12);
        let gumbel = -(-u.ln()).ln();
        let score = scaled + gumbel;
        if score > best_score {
            best_score = score;
            best_token = idx;
        }
    }

    Ok(best_token)
}

/// Detect cyclical token repetition at the tail of `history` (periods 1..=12),
/// borrowed from `spark-tts-rs` (`check_token_repetition`).
pub fn check_semantic_repetition(history: &[i64]) -> bool {
    let len = history.len();
    for n in 1..=12 {
        let min_repeats = if n == 1 {
            8
        } else if n <= 3 {
            5
        } else {
            4
        };
        if len >= n * min_repeats {
            let pattern = &history[len - n..];
            let mut all_match = true;
            for r in 1..min_repeats {
                let start = len - (r + 1) * n;
                if &history[start..start + n] != pattern {
                    all_match = false;
                    break;
                }
            }
            if all_match {
                return true;
            }
        }
    }
    false
}

/// Sliding-window streaming vocoder with guard frames and linear crossfade.
/// Combines `Audio8_TTS/onnx_runtime/arktts_runtime/runtime.py` (`stream()`)
/// with `spark-tts-rs` (`ProgressiveStreamer`).
pub struct ProgressiveAudio8Streamer {
    all_frames: Vec<Vec<i64>>,
    emitted_samples: usize,
    hop_length: usize,
    context_frames: usize,
    guard_samples: usize,
    first_chunk_frames: usize,
    chunk_frames: usize,
    first_emitted: bool,
    crossfade_tail: Vec<f32>,
    crossfade_len: usize,
}

impl ProgressiveAudio8Streamer {
    pub fn new(manifest: &RuntimeManifest) -> Self {
        let hop = manifest.codec_hop_length.max(1);
        let context = manifest.stream_context_frames.max(16);
        let guard = manifest.stream_guard_frames * hop;
        let crossfade_len = ((manifest.sample_rate as usize) / 100).max(64);
        Self {
            all_frames: Vec::with_capacity(256),
            emitted_samples: 0,
            hop_length: hop,
            context_frames: context,
            guard_samples: guard,
            first_chunk_frames: 8,
            chunk_frames: 12,
            first_emitted: false,
            crossfade_tail: Vec::new(),
            crossfade_len,
        }
    }

    pub fn frame_count(&self) -> usize {
        self.all_frames.len()
    }

    pub fn push_frame(
        &mut self,
        frame: Vec<i64>,
        engine: &mut Audio8OnnxEngine,
    ) -> Result<Option<Vec<f32>>> {
        self.all_frames.push(frame);
        let total = self.all_frames.len();

        let should_decode = if !self.first_emitted {
            total >= self.first_chunk_frames
        } else {
            (total - self.first_chunk_frames) % self.chunk_frames == 0
        };

        if !should_decode {
            return Ok(None);
        }

        self.decode_window(engine, false)
    }

    pub fn finalize(&mut self, engine: &mut Audio8OnnxEngine) -> Result<Option<Vec<f32>>> {
        if self.all_frames.is_empty() {
            return Ok(None);
        }
        self.decode_window(engine, true)
    }

    fn decode_window(
        &mut self,
        engine: &mut Audio8OnnxEngine,
        is_final: bool,
    ) -> Result<Option<Vec<f32>>> {
        let total_frames = self.all_frames.len();
        let num_cb = engine.manifest.num_codebooks;
        let window_span = self.context_frames + self.chunk_frames;
        let start_frame = total_frames.saturating_sub(window_span);
        let window_frames = total_frames - start_frame;

        let mut window_codes = Array2::<i64>::zeros((num_cb, window_frames));
        for (col, frame) in self.all_frames[start_frame..].iter().enumerate() {
            for (row, &val) in frame.iter().take(num_cb).enumerate() {
                window_codes[[row, col]] = val;
            }
        }

        let audio = engine.decode_codes(&window_codes)?;
        let absolute_start = start_frame * self.hop_length;
        let guard = if is_final { 0 } else { self.guard_samples };
        let stable_end = absolute_start + audio.len().saturating_sub(guard);

        let begin = self.emitted_samples.saturating_sub(absolute_start);
        let end = stable_end
            .saturating_sub(absolute_start)
            .max(begin)
            .min(audio.len());

        if end <= begin {
            return Ok(None);
        }

        let mut chunk = audio[begin..end].to_vec();
        self.emitted_samples += chunk.len();

        if !self.crossfade_tail.is_empty() && chunk.len() >= self.crossfade_tail.len() {
            let n = self.crossfade_tail.len();
            for i in 0..n {
                let w = (i as f32 + 1.0) / (n as f32 + 1.0);
                chunk[i] = self.crossfade_tail[i] * (1.0 - w) + chunk[i] * w;
            }
            self.crossfade_tail.clear();
        }

        if !is_final && chunk.len() > self.crossfade_len * 2 {
            let split = chunk.len() - self.crossfade_len;
            self.crossfade_tail = chunk[split..].to_vec();
            chunk.truncate(split);
            self.emitted_samples -= self.crossfade_len;
        }

        if !self.first_emitted {
            let fade_samples = ((engine.manifest.sample_rate as usize) * 5) / 1000;
            audio::apply_fade_in(&mut chunk, fade_samples);
            self.first_emitted = true;
        }

        if is_final {
            let fade_samples = ((engine.manifest.sample_rate as usize) * 5) / 1000;
            audio::apply_fade_out(&mut chunk, fade_samples);
        }

        Ok(Some(chunk))
    }
}

pub enum SlowCacheState {
    PerLayer06B {
        keys: Vec<Array4<f32>>,
        values: Vec<Array4<f32>>,
    },
}

pub struct Audio8OnnxEngine {
    pub manifest: RuntimeManifest,
    pub precision: String,
    pub codec_precision: String,
    pub active_provider: String,
    slow: Session,
    fast: Session,
    decoder: Session,
    encoder: Option<Session>,
    slow_single_token_only: bool,
}

impl Audio8OnnxEngine {
    pub fn load(
        model_dir: &Path,
        precision: Option<&str>,
        codec_precision: Option<&str>,
        device: ExecutionDevice,
        threads: usize,
    ) -> Result<Self> {
        let manifest = RuntimeManifest::load_or_default(model_dir)?;
        let prec = precision
            .unwrap_or(&manifest.default_precision)
            .to_string();
        let codec_prec = codec_precision
            .unwrap_or(&manifest.default_codec_precision)
            .to_string();

        let slow_path = manifest.resolve_slow_model(model_dir, &prec);
        let fast_path = manifest.resolve_fast_model(model_dir, &prec);
        let decoder_path = manifest.resolve_codec_decoder_model(model_dir, &codec_prec);
        let encoder_path = manifest.resolve_codec_encoder_model(model_dir);

        println!(
            "Loading Audio8 ONNX sessions from {} (precision={}, codec={}, device={:?})...",
            model_dir.display(),
            prec,
            codec_prec,
            device
        );

        let (slow, slow_ep) = build_ort_session(&slow_path, device, threads)
            .with_context(|| format!("Failed to load Slow AR ONNX: {}", slow_path.display()))?;
        let (fast, _fast_ep) = build_ort_session(&fast_path, device, threads)
            .with_context(|| format!("Failed to load Fast AR ONNX: {}", fast_path.display()))?;
        let (decoder, _dec_ep) = build_ort_session(&decoder_path, device, threads)
            .with_context(|| {
                format!(
                    "Failed to load Codec Decoder ONNX: {}",
                    decoder_path.display()
                )
            })?;

        let encoder = if let Some(enc_path) = encoder_path {
            match build_ort_session(&enc_path, device, threads) {
                Ok((sess, _)) => {
                    println!("Loaded Codec Encoder ONNX: {}", enc_path.display());
                    Some(sess)
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to load optional Codec Encoder {}: {}",
                        enc_path.display(),
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        let mut slow_single_token_only = false;
        for inp in slow.inputs() {
            if inp.name() == "codes" {
                if let ValueType::Tensor { shape, .. } = inp.dtype() {
                    if shape.len() == 3 && shape[2] == 1 {
                        slow_single_token_only = true;
                    }
                }
            }
        }

        println!(
            "Audio8 ONNX Runtime ready on provider: {} (slow_single_token_only={})",
            slow_ep, slow_single_token_only
        );

        Ok(Self {
            manifest,
            precision: prec,
            codec_precision: codec_prec,
            active_provider: slow_ep,
            slow,
            fast,
            decoder,
            encoder,
            slow_single_token_only,
        })
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    fn init_slow_caches(&self) -> SlowCacheState {
        let num_layers = self.manifest.num_layers;
        let heads = self.manifest.n_local_heads;
        let max_seq = self.manifest.max_seq_len;
        let head_dim = self.manifest.head_dim;

        let keys = (0..num_layers)
            .map(|_| Array4::<f32>::zeros((1, heads, max_seq, head_dim)))
            .collect();
        let values = (0..num_layers)
            .map(|_| Array4::<f32>::zeros((1, heads, max_seq, head_dim)))
            .collect();
        SlowCacheState::PerLayer06B { keys, values }
    }

    fn init_fast_caches(&self) -> (Vec<Array4<f32>>, Vec<Array4<f32>>) {
        let num_fast = self.manifest.num_fast_layers;
        let heads = self.manifest.n_local_heads;
        let max_cb = self.manifest.num_codebooks;
        let fast_dim = self.manifest.fast_head_dim;

        let keys = (0..num_fast)
            .map(|_| Array4::<f32>::zeros((1, heads, max_cb, fast_dim)))
            .collect();
        let values = (0..num_fast)
            .map(|_| Array4::<f32>::zeros((1, heads, max_cb, fast_dim)))
            .collect();
        (keys, values)
    }

    fn slow_step(
        &mut self,
        codes: &Array3<i64>,
        start_pos: usize,
        state: &mut SlowCacheState,
    ) -> Result<(Vec<f32>, Array3<f32>)> {
        let seq_len = codes.shape()[2];
        let input_pos: Array1<i64> = (start_pos..start_pos + seq_len)
            .map(|p| p as i64)
            .collect();

        let SlowCacheState::PerLayer06B { keys, values } = state;
        let num_layers = self.manifest.num_layers;

        let mut feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(2 + num_layers * 2);
        feeds.push((
            Cow::Borrowed("codes"),
            Tensor::from_array(codes.clone())?.into(),
        ));
        feeds.push((
            Cow::Borrowed("input_pos"),
            Tensor::from_array(input_pos)?.into(),
        ));

        for i in 0..num_layers {
            feeds.push((
                Cow::Owned(format!("cache_key_{}", i)),
                Tensor::from_array(keys[i].clone())?.into(),
            ));
            feeds.push((
                Cow::Owned(format!("cache_value_{}", i)),
                Tensor::from_array(values[i].clone())?.into(),
            ));
        }

        let outputs = self
            .slow
            .run(feeds)
            .map_err(|e| anyhow::anyhow!("Slow AR ONNX run failed at pos {}: {}", start_pos, e))?;

        let logits_view = outputs["logits"]
            .try_extract_array::<f32>()
            .context("Failed to extract Slow AR logits as f32")?;
        let logits_shape = logits_view.shape();
        let vocab_dim = *logits_shape.last().unwrap_or(&0);
        let logits_flat = logits_view.as_slice().context("Non-contiguous logits")?;
        let last_logits = logits_flat[logits_flat.len() - vocab_dim..].to_vec();

        let hidden_view = outputs["hidden"]
            .try_extract_array::<f32>()
            .context("Failed to extract Slow AR hidden as f32")?;
        let dim = *hidden_view.shape().last().unwrap_or(&self.manifest.dim);
        let hidden_flat = hidden_view.as_slice().context("Non-contiguous hidden")?;
        let last_hidden_slice = &hidden_flat[hidden_flat.len() - dim..];
        let last_hidden = Array3::from_shape_vec((1, 1, dim), last_hidden_slice.to_vec())?;

        for i in 0..num_layers {
            let k_name = format!("next_cache_key_{}", i);
            let v_name = format!("next_cache_value_{}", i);
            if let Some(k_out) = outputs.get(&k_name) {
                let k_arr = k_out.try_extract_array::<f32>()?;
                update_cache_4d(&mut keys[i], &k_arr, start_pos)?;
            }
            if let Some(v_out) = outputs.get(&v_name) {
                let v_arr = v_out.try_extract_array::<f32>()?;
                update_cache_4d(&mut values[i], &v_arr, start_pos)?;
            }
        }

        Ok((last_logits, last_hidden))
    }

    fn fast_step(
        &mut self,
        slow_hidden: &Array3<f32>,
        token_id: i64,
        use_slow_hidden: bool,
        position: usize,
        fast_keys: &mut [Array4<f32>],
        fast_values: &mut [Array4<f32>],
    ) -> Result<Vec<f32>> {
        let num_fast = self.manifest.num_fast_layers;
        let mut feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(4 + num_fast * 2);

        let token_arr = Array2::from_elem((1, 1), token_id);
        let use_slow_arr = Array1::from_elem(1, use_slow_hidden);
        let pos_arr = Array1::from_elem(1, position as i64);

        feeds.push((
            Cow::Borrowed("slow_hidden"),
            Tensor::from_array(slow_hidden.clone())?.into(),
        ));
        feeds.push((
            Cow::Borrowed("token_id"),
            Tensor::from_array(token_arr)?.into(),
        ));
        feeds.push((
            Cow::Borrowed("use_slow_hidden"),
            Tensor::from_array(use_slow_arr)?.into(),
        ));
        feeds.push((
            Cow::Borrowed("input_pos"),
            Tensor::from_array(pos_arr)?.into(),
        ));

        for i in 0..num_fast {
            feeds.push((
                Cow::Owned(format!("cache_key_{}", i)),
                Tensor::from_array(fast_keys[i].clone())?.into(),
            ));
            feeds.push((
                Cow::Owned(format!("cache_value_{}", i)),
                Tensor::from_array(fast_values[i].clone())?.into(),
            ));
        }

        let outputs = self
            .fast
            .run(feeds)
            .map_err(|e| anyhow::anyhow!("Fast AR ONNX run failed at pos {}: {}", position, e))?;

        let logits_view = outputs["logits"]
            .try_extract_array::<f32>()
            .context("Failed to extract Fast AR logits as f32")?;
        let cb_size = *logits_view.shape().last().unwrap_or(&self.manifest.codebook_size);
        let logits_flat = logits_view.as_slice().context("Non-contiguous fast logits")?;
        let last_logits = logits_flat[logits_flat.len() - cb_size..].to_vec();

        for i in 0..num_fast {
            let k_name = format!("next_cache_key_{}", i);
            let v_name = format!("next_cache_value_{}", i);
            if let Some(k_out) = outputs.get(&k_name) {
                let k_arr = k_out.try_extract_array::<f32>()?;
                update_cache_4d(&mut fast_keys[i], &k_arr, position)?;
            }
            if let Some(v_out) = outputs.get(&v_name) {
                let v_arr = v_out.try_extract_array::<f32>()?;
                update_cache_4d(&mut fast_values[i], &v_arr, position)?;
            }
        }

        Ok(last_logits)
    }

    fn sample_semantic(
        &self,
        logits: &[f32],
        previous: &[i64],
        temperature: f32,
        top_p: f32,
        top_k: usize,
        rng: &mut StdRng,
    ) -> Result<i64> {
        let begin = self.manifest.semantic_begin_id;
        let end = self.manifest.semantic_end_id;
        let stop = self.manifest.im_end_id;
        let cb_size = self.manifest.codebook_size;
        let expected = cb_size + 1;

        let mut values = if logits.len() == expected {
            logits.to_vec()
        } else {
            let mut packed = Vec::with_capacity(expected);
            for id in begin..=end {
                let idx = id as usize;
                packed.push(if idx < logits.len() {
                    logits[idx]
                } else {
                    f32::NEG_INFINITY
                });
            }
            let eos_idx = stop as usize;
            packed.push(if eos_idx < logits.len() {
                logits[eos_idx]
            } else {
                f32::NEG_INFINITY
            });
            packed
        };

        let eos_first = self.manifest.slow_logits_layout == "eos_then_relative_semantic";
        let eos_index = if eos_first { 0 } else { expected - 1 };

        if previous.is_empty() && eos_index < values.len() {
            values[eos_index] = f32::NEG_INFINITY;
        }

        let normal_idx = sample_logits(&values, temperature, top_p, top_k, rng)?;
        let high_idx = sample_logits(
            &values,
            self.manifest.ras_temperature,
            self.manifest.ras_top_p,
            top_k,
            rng,
        )?;

        let map_index = |idx: usize| -> i64 {
            if idx == eos_index {
                stop
            } else if eos_first {
                begin + (idx as i64 - 1)
            } else {
                begin + (idx as i64)
            }
        };

        let normal = map_index(normal_idx);
        let high = map_index(high_idx);

        if normal >= begin && normal <= end && previous.contains(&normal) {
            Ok(high)
        } else {
            Ok(normal)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_frames_with_callback<F>(
        &mut self,
        prompt: &Array3<i64>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
        abort_flag: Option<&Arc<AtomicBool>>,
        mut on_frame: F,
    ) -> Result<usize>
    where
        F: FnMut(Vec<i64>, &mut Self) -> Result<bool>,
    {
        let prompt_len = prompt.shape()[2];
        let max_seq_len = self.manifest.max_seq_len;
        if prompt_len >= max_seq_len {
            anyhow::bail!(
                "Prompt length {} exceeds max_seq_len {}",
                prompt_len,
                max_seq_len
            );
        }
        let max_steps = max_new_tokens.min(max_seq_len - prompt_len);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut slow_state = self.init_slow_caches();

        let mut logits = Vec::new();
        let mut hidden = Array3::<f32>::zeros((1, 1, self.manifest.dim));

        if self.slow_single_token_only {
            for pos in 0..prompt_len {
                if let Some(flag) = abort_flag {
                    if flag.load(Ordering::Relaxed) {
                        return Ok(0);
                    }
                }
                let col = prompt.slice(s![.., .., pos..pos + 1]).to_owned();
                let (l, h) = self.slow_step(&col, pos, &mut slow_state)?;
                logits = l;
                hidden = h;
            }
        } else {
            let chunk_size = 128;
            let mut offset = 0;
            while offset < prompt_len {
                if let Some(flag) = abort_flag {
                    if flag.load(Ordering::Relaxed) {
                        return Ok(0);
                    }
                }
                let end = (offset + chunk_size).min(prompt_len);
                let chunk = prompt.slice(s![.., .., offset..end]).to_owned();
                let (l, h) = self.slow_step(&chunk, offset, &mut slow_state)?;
                logits = l;
                hidden = h;
                offset = end;
            }
        }

        let mut ras_window: Vec<i64> = Vec::with_capacity(self.manifest.ras_window_size + 1);
        let mut full_semantic_history: Vec<i64> = Vec::with_capacity(max_steps);
        let begin = self.manifest.semantic_begin_id;
        let stop = self.manifest.im_end_id;
        let num_cb = self.manifest.num_codebooks;
        let cb_size = self.manifest.codebook_size as i64;
        let mut generated_frames = 0;

        for step in 0..max_steps {
            if let Some(flag) = abort_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let semantic =
                self.sample_semantic(&logits, &ras_window, temperature, top_p, top_k, &mut rng)?;
            if semantic == stop {
                break;
            }

            full_semantic_history.push(semantic);
            if check_semantic_repetition(&full_semantic_history) {
                println!(
                    "Detected cyclical semantic token repetition at frame {}. Stopping segment cleanly.",
                    step
                );
                break;
            }

            ras_window.push(semantic);
            if ras_window.len() > self.manifest.ras_window_size {
                ras_window.remove(0);
            }

            let first_code = (semantic - begin).clamp(0, cb_size - 1);
            let mut frame = Vec::with_capacity(num_cb);
            frame.push(first_code);

            let (mut fast_keys, mut fast_values) = self.init_fast_caches();
            self.fast_step(&hidden, 0, true, 0, &mut fast_keys, &mut fast_values)?;

            let mut prev_code = first_code;
            for fast_pos in 1..num_cb {
                let fast_logits = self.fast_step(
                    &hidden,
                    prev_code,
                    false,
                    fast_pos,
                    &mut fast_keys,
                    &mut fast_values,
                )?;
                let next_code =
                    sample_logits(&fast_logits, temperature, top_p, top_k, &mut rng)? as i64;
                frame.push(next_code);
                prev_code = next_code;
            }

            generated_frames += 1;

            let mut next_col = Array3::<i64>::zeros((1, num_cb + 1, 1));
            next_col[[0, 0, 0]] = semantic;
            for (cb_idx, &code) in frame.iter().enumerate() {
                next_col[[0, cb_idx + 1, 0]] = code;
            }

            let keep_going = on_frame(frame, self)?;
            if !keep_going || step + 1 >= max_steps {
                break;
            }

            let (next_logits, next_hidden) =
                self.slow_step(&next_col, prompt_len + step, &mut slow_state)?;
            logits = next_logits;
            hidden = next_hidden;
        }

        Ok(generated_frames)
    }

    pub fn decode_codes(&mut self, codes: &Array2<i64>) -> Result<Vec<f32>> {
        let (rows, cols) = codes.dim();
        if rows != self.manifest.num_codebooks || cols == 0 {
            anyhow::bail!(
                "Invalid codec input shape [{}, {}], expected [{}, T]",
                rows,
                cols,
                self.manifest.num_codebooks
            );
        }

        let input_3d = codes.clone().insert_axis(ndarray::Axis(0));
        let input_name = self
            .decoder
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "codes".to_string());

        let feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![(
            Cow::Owned(input_name),
            Tensor::from_array(input_3d)?.into(),
        )];

        let outputs = self
            .decoder
            .run(feeds)
            .map_err(|e| anyhow::anyhow!("Codec Decoder ONNX run failed: {}", e))?;

        let audio_view = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract decoded waveform as f32")?;
        Ok(audio_view.iter().copied().collect())
    }

    pub fn encode_reference_audio(&mut self, samples_44k: &[f32]) -> Result<Array2<i64>> {
        let prepared = audio::prepare_reference_audio(samples_44k, self.manifest.codec_hop_length)?;
        let encoder = self.encoder.as_mut().context(
            "Codec encoder ONNX model (codec_encoder.onnx) is not loaded; cannot encode raw reference audio",
        )?;

        let input_name = encoder
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "audio".to_string());
        let rank = encoder
            .inputs()
            .first()
            .and_then(|i| match i.dtype() {
                ValueType::Tensor { shape, .. } => Some(shape.len()),
                _ => None,
            })
            .unwrap_or(3);

        let feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = if rank == 2 {
            let arr = Array2::from_shape_vec((1, prepared.len()), prepared)?;
            vec![(Cow::Owned(input_name), Tensor::from_array(arr)?.into())]
        } else {
            let arr = Array3::from_shape_vec((1, 1, prepared.len()), prepared)?;
            vec![(Cow::Owned(input_name), Tensor::from_array(arr)?.into())]
        };

        let outputs = encoder
            .run(feeds)
            .map_err(|e| anyhow::anyhow!("Codec Encoder ONNX run failed: {}", e))?;

        let codes_view = outputs[0]
            .try_extract_array::<i64>()
            .context("Failed to extract encoded reference codes as i64")?;
        let shape = codes_view.shape();
        let (rows, cols) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            anyhow::bail!("Unexpected encoder output shape: {:?}", shape);
        };

        if rows != self.manifest.num_codebooks || cols == 0 {
            anyhow::bail!(
                "Codec encoder returned shape {:?}, expected [{}, T]",
                shape,
                self.manifest.num_codebooks
            );
        }

        let flat: Vec<i64> = codes_view.iter().copied().collect();
        Array2::from_shape_vec((rows, cols), flat).context("Failed to reshape encoded VQ codes")
    }
}

fn update_cache_4d(
    cache: &mut Array4<f32>,
    value: &ndarray::ArrayViewD<'_, f32>,
    position: usize,
) -> Result<()> {
    let c_shape = cache.shape().to_vec();
    let v_shape = value.shape();
    if v_shape == c_shape.as_slice() {
        let v_4d = value
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .context("Cache shape mismatch")?;
        cache.assign(&v_4d);
        return Ok(());
    }
    if v_shape.len() == 4
        && v_shape[0] == c_shape[0]
        && v_shape[1] == c_shape[1]
        && v_shape[3] == c_shape[3]
    {
        let len = v_shape[2];
        if position + len <= c_shape[2] {
            let v_4d = value
                .view()
                .into_dimensionality::<ndarray::Ix4>()
                .context("Cache delta dimensionality mismatch")?;
            cache
                .slice_mut(s![.., .., position..position + len, ..])
                .assign(&v_4d);
            return Ok(());
        }
    }
    anyhow::bail!(
        "Unexpected KV cache output shape {:?} for cache {:?} at position {}",
        v_shape,
        c_shape,
        position
    );
}
