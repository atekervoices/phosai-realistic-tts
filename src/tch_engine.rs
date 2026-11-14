use anyhow::{Context, Result};
use ndarray::{s, Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tch::{CModule, Device, IValue, Kind, Tensor};

use crate::audio;
use crate::config::{ExecutionDevice, RuntimeManifest};

/// Top-K / Top-P / Temperature Gumbel-Max sampler for Slow AR semantic tokens.
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
/// matching `spark-tts-rs` (`check_token_repetition`). Ignores first 40 frames (~1.8s)
/// so initial acoustic pause/lead-in is never falsely flagged as a loop.
pub fn check_semantic_repetition(history: &[i64]) -> bool {
    let len = history.len();
    if len < 40 {
        return false;
    }
    for n in 1..=12 {
        let min_repeats = if n == 1 {
            16
        } else if n <= 3 {
            6
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
            first_chunk_frames: 2,
            chunk_frames: 3,
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
        engine: &mut Audio8TchEngine,
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

    pub fn finalize(&mut self, engine: &mut Audio8TchEngine) -> Result<Option<Vec<f32>>> {
        if self.all_frames.is_empty() {
            return Ok(None);
        }
        self.decode_window(engine, true)
    }

    fn decode_window(
        &mut self,
        engine: &mut Audio8TchEngine,
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

/// Native LibTorch (`tch-rs`) inference worker holding SlowAR, fused FastARLoop,
/// CodecDecoder, and optional CodecEncoder TorchScript modules on CUDA or CPU.
pub struct Audio8TchEngine {
    pub manifest: RuntimeManifest,
    pub precision: String,
    pub codec_precision: String,
    pub active_provider: String,
    pub device: Device,
    pub kv_kind: Kind,
    slow: CModule,
    fast: CModule,
    decoder: CModule,
    encoder: Option<CModule>,
}

// Each Audio8TchEngine instance is guarded by an exclusive Mutex in TtsEngine's worker pool,
// and executes only immutable eval-mode forward passes under no_grad_guard.
unsafe impl Send for Audio8TchEngine {}
unsafe impl Sync for Audio8TchEngine {}

impl Audio8TchEngine {
    pub fn load(
        model_dir: &Path,
        precision: Option<&str>,
        codec_precision: Option<&str>,
        device: ExecutionDevice,
        _threads: usize,
    ) -> Result<Self> {
        let _no_grad = tch::no_grad_guard();
        unsafe {
            torch_sys::cuda::atc_set_benchmark_cudnn(0);
        }

        let has_cuda = tch::Cuda::is_available();
        let (tch_device, active_provider) = match device {
            ExecutionDevice::Cpu => (Device::Cpu, "LibTorchCPU".to_string()),
            ExecutionDevice::Gpu => {
                if !has_cuda {
                    anyhow::bail!(
                        "GPU execution requested (--device gpu), but LibTorch CUDA is unavailable."
                    );
                }
                (Device::Cuda(0), "LibTorchCUDA".to_string())
            }
            ExecutionDevice::Auto => {
                if has_cuda {
                    (Device::Cuda(0), "LibTorchCUDA".to_string())
                } else {
                    (Device::Cpu, "LibTorchCPU".to_string())
                }
            }
        };

        let is_cuda = matches!(tch_device, Device::Cuda(_));
        let manifest = RuntimeManifest::load_or_default(model_dir)?;

        let mut req_prec = precision
            .unwrap_or(&manifest.default_precision)
            .to_lowercase();
        if !is_cuda && precision.is_none() && manifest.resolve_slow_model(model_dir, "fp32").is_file()
        {
            req_prec = "fp32".to_string();
        }

        let slow_path = manifest.resolve_slow_model(model_dir, &req_prec);
        if !slow_path.is_file() {
            anyhow::bail!(
                "Slow AR TorchScript model not found at {} (run `python scripts/export_torchscript.py`)",
                slow_path.display()
            );
        }
        let slow_is_fp32 = slow_path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| n.contains("fp32"));
        let effective_prec = if slow_is_fp32 { "fp32" } else { "fp16" }.to_string();
        let kv_kind = if slow_is_fp32 {
            Kind::Float
        } else {
            Kind::Half
        };

        let fast_path = manifest.resolve_fast_model(model_dir, &effective_prec);
        if !fast_path.is_file() {
            anyhow::bail!(
                "Fast AR TorchScript model not found at {}",
                fast_path.display()
            );
        }

        let mut req_codec_prec = codec_precision
            .unwrap_or(&manifest.default_codec_precision)
            .to_lowercase();
        if !is_cuda
            && codec_precision.is_none()
            && manifest
                .resolve_codec_decoder_model(model_dir, "fp32")
                .is_file()
        {
            req_codec_prec = "fp32".to_string();
        }

        let decoder_path = manifest.resolve_codec_decoder_model(model_dir, &req_codec_prec);
        if !decoder_path.is_file() {
            anyhow::bail!(
                "Codec Decoder TorchScript model not found at {}",
                decoder_path.display()
            );
        }
        let effective_codec_prec = if decoder_path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| n.contains("fp16"))
        {
            "fp16".to_string()
        } else {
            "fp32".to_string()
        };

        println!(
            "Loading Audio8 LibTorch modules on {} ({}):\n  Slow AR : {} ({})\n  Fast AR : {}\n  Decoder : {} ({})",
            active_provider,
            if is_cuda { "CUDA:0" } else { "CPU" },
            slow_path.display(),
            effective_prec,
            fast_path.display(),
            decoder_path.display(),
            effective_codec_prec,
        );

        let mut slow = CModule::load_on_device(&slow_path, tch_device)
            .with_context(|| format!("Failed to load SlowAR {}", slow_path.display()))?;
        slow.set_eval();

        let mut fast = CModule::load_on_device(&fast_path, tch_device)
            .with_context(|| format!("Failed to load FastAR {}", fast_path.display()))?;
        fast.set_eval();

        let mut decoder = CModule::load_on_device(&decoder_path, tch_device)
            .with_context(|| format!("Failed to load CodecDecoder {}", decoder_path.display()))?;
        decoder.set_eval();

        let encoder = if let Some(enc_path) = manifest.resolve_codec_encoder_model(model_dir) {
            println!("  Encoder : {}", enc_path.display());
            let mut enc = CModule::load_on_device(&enc_path, tch_device)
                .with_context(|| format!("Failed to load CodecEncoder {}", enc_path.display()))?;
            enc.set_eval();
            Some(enc)
        } else {
            None
        };

        let mut engine = Self {
            manifest,
            precision: effective_prec,
            codec_precision: effective_codec_prec,
            active_provider,
            device: tch_device,
            kv_kind,
            slow,
            fast,
            decoder,
            encoder,
        };
        engine.warmup()?;

        Ok(engine)
    }

    fn alloc_slow_kv(&self) -> (Tensor, Tensor) {
        let shape = [
            self.manifest.num_layers as i64,
            1,
            self.manifest.n_local_heads as i64,
            self.manifest.max_seq_len as i64,
            self.manifest.head_dim as i64,
        ];
        (
            Tensor::zeros(shape, (self.kv_kind, self.device)),
            Tensor::zeros(shape, (self.kv_kind, self.device)),
        )
    }

    /// Pre-warm cuBLAS / cuDNN workspaces and TorchScript JIT execution plans at startup
    /// so the very first HTTP/WS request suffers zero cold-start stall.
    pub fn warmup(&mut self) -> Result<()> {
        let _no_grad = tch::no_grad_guard();
        let num_cb = self.manifest.num_codebooks;
        let (k_caches, v_caches) = self.alloc_slow_kv();

        let dummy_prefill = Array3::<i64>::zeros((1, num_cb + 1, 8));
        let (_, hidden_gpu) = self.slow_step(&dummy_prefill, 0, &k_caches, &v_caches)?;

        let dummy_step = Array3::<i64>::zeros((1, num_cb + 1, 1));
        let _ = self.slow_step(&dummy_step, 8, &k_caches, &v_caches)?;

        let first_code_t = Tensor::from_slice(&[0i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let temp_t = Tensor::from_slice(&[0.7f32])
            .to_kind(Kind::Float)
            .to_device(self.device);
        let top_p_t = Tensor::from_slice(&[0.8f32])
            .to_kind(Kind::Float)
            .to_device(self.device);
        let top_k_t = Tensor::from_slice(&[20i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let _ = self.fast.forward_ts(&[
            hidden_gpu,
            first_code_t,
            temp_t,
            top_p_t,
            top_k_t,
        ])?;

        for warm_frames in [2usize, 8usize] {
            let dummy_vq = Array2::<i64>::zeros((num_cb, warm_frames));
            let _ = self.decode_codes(&dummy_vq)?;
        }

        println!("LibTorch worker warmup complete ({}).", self.active_provider);
        Ok(())
    }

    fn slow_step(
        &mut self,
        codes: &Array3<i64>,
        start_pos: usize,
        k_caches: &Tensor,
        v_caches: &Tensor,
    ) -> Result<(Vec<f32>, Tensor)> {
        let rows = codes.shape()[1];
        let seq_len = codes.shape()[2];
        let codes_std = codes.as_standard_layout();
        let codes_slice = codes_std
            .as_slice()
            .context("Non-contiguous SlowAR input codes")?;

        let codes_t = Tensor::from_slice(codes_slice)
            .reshape(&[1, rows as i64, seq_len as i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);

        let pos_vec: Vec<i64> = (start_pos..start_pos + seq_len)
            .map(|p| p as i64)
            .collect();
        let pos_t = Tensor::from_slice(&pos_vec)
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let valid_len_t = Tensor::from_slice(&[(start_pos + seq_len) as i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);

        let iv = self
            .slow
            .forward_is(&[
                IValue::from(codes_t),
                IValue::from(pos_t),
                IValue::from(valid_len_t),
                IValue::from(k_caches.shallow_clone()),
                IValue::from(v_caches.shallow_clone()),
            ])
            .map_err(|e| anyhow::anyhow!("SlowAR forward failed at pos {}: {}", start_pos, e))?;

        let (logits_t, fast_hidden) = match iv {
            IValue::Tuple(mut items) if items.len() == 2 => {
                let hidden_iv = items.pop().unwrap();
                let logits_iv = items.pop().unwrap();
                match (logits_iv, hidden_iv) {
                    (IValue::Tensor(l), IValue::Tensor(h)) => (l, h),
                    _ => anyhow::bail!("SlowAR returned non-Tensor elements inside Tuple"),
                }
            }
            _ => anyhow::bail!("SlowAR returned unexpected non-Tuple IValue"),
        };

        let logits_cpu = logits_t
            .flatten(0, -1)
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
        let n = logits_cpu.numel();
        let mut logits = vec![0.0f32; n as usize];
        logits_cpu.copy_data(&mut logits, n);

        Ok((logits, fast_hidden))
    }

    fn sample_semantic(
        &self,
        logits: &[f32],
        previous: &[i64],
        step: usize,
        min_frames: usize,
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

        if (step < min_frames || previous.is_empty()) && eos_index < values.len() {
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
        let _no_grad = tch::no_grad_guard();

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
        let (k_caches, v_caches) = self.alloc_slow_kv();

        let mut logits = Vec::new();
        let mut hidden_gpu: Option<Tensor> = None;

        let chunk_size = 256;
        let mut offset = 0;
        while offset < prompt_len {
            if let Some(flag) = abort_flag {
                if flag.load(Ordering::Relaxed) {
                    return Ok(0);
                }
            }
            let end = (offset + chunk_size).min(prompt_len);
            let chunk = prompt.slice(s![.., .., offset..end]).to_owned();
            let (l, h) = self.slow_step(&chunk, offset, &k_caches, &v_caches)?;
            logits = l;
            hidden_gpu = Some(h);
            offset = end;
        }

        let mut current_hidden = hidden_gpu.context("Prompt prefill produced no hidden state")?;

        let temp_t = Tensor::from_slice(&[temperature])
            .to_kind(Kind::Float)
            .to_device(self.device);
        let top_p_t = Tensor::from_slice(&[top_p])
            .to_kind(Kind::Float)
            .to_device(self.device);
        let top_k_t = Tensor::from_slice(&[top_k as i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);

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

            let semantic = self.sample_semantic(
                &logits,
                &ras_window,
                step,
                16,
                temperature,
                top_p,
                top_k,
                &mut rng,
            )?;
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
            let first_code_t = Tensor::from_slice(&[first_code])
                .to_kind(Kind::Int64)
                .to_device(self.device);

            let frame_t = self
                .fast
                .forward_ts(&[
                    current_hidden.shallow_clone(),
                    first_code_t,
                    temp_t.shallow_clone(),
                    top_p_t.shallow_clone(),
                    top_k_t.shallow_clone(),
                ])
                .map_err(|e| anyhow::anyhow!("FastAR loop failed at step {}: {}", step, e))?;

            let frame_cpu = frame_t
                .flatten(0, -1)
                .to_kind(Kind::Int64)
                .to_device(Device::Cpu);
            let n = frame_cpu.numel();
            let mut frame = vec![0i64; n as usize];
            frame_cpu.copy_data(&mut frame, n);

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
                self.slow_step(&next_col, prompt_len + step, &k_caches, &v_caches)?;
            logits = next_logits;
            current_hidden = next_hidden;
        }

        Ok(generated_frames)
    }

    pub fn decode_codes(&mut self, codes: &Array2<i64>) -> Result<Vec<f32>> {
        let _no_grad = tch::no_grad_guard();
        let (rows, cols) = codes.dim();
        if rows != self.manifest.num_codebooks || cols == 0 {
            anyhow::bail!(
                "Invalid codec input shape [{}, {}], expected [{}, T]",
                rows,
                cols,
                self.manifest.num_codebooks
            );
        }

        let codes_std = codes.as_standard_layout();
        let codes_slice = codes_std
            .as_slice()
            .context("Non-contiguous codec decoder input")?;
        let codes_t = Tensor::from_slice(codes_slice)
            .reshape(&[1, rows as i64, cols as i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);

        let audio_t = self
            .decoder
            .forward_ts(&[codes_t])
            .map_err(|e| anyhow::anyhow!("CodecDecoder forward failed: {}", e))?;

        let audio_cpu = audio_t
            .flatten(0, -1)
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
        let n = audio_cpu.numel();
        let mut pcm = vec![0.0f32; n as usize];
        audio_cpu.copy_data(&mut pcm, n);

        Ok(pcm)
    }

    pub fn encode_reference_audio(&mut self, samples_44k: &[f32]) -> Result<Array2<i64>> {
        let _no_grad = tch::no_grad_guard();
        let prepared = audio::prepare_reference_audio(samples_44k, self.manifest.codec_hop_length)?;
        let encoder = self.encoder.as_ref().context(
            "Codec encoder TorchScript model (codec_encoder_fp32.pt) is not loaded; cannot encode raw reference audio",
        )?;

        let audio_t = Tensor::from_slice(&prepared)
            .reshape(&[1, 1, prepared.len() as i64])
            .to_kind(Kind::Float)
            .to_device(self.device);

        let codes_t = encoder
            .forward_ts(&[audio_t])
            .map_err(|e| anyhow::anyhow!("CodecEncoder forward failed: {}", e))?;

        let shape = codes_t.size();
        let (rows, cols) = if shape.len() == 3 {
            (shape[1] as usize, shape[2] as usize)
        } else if shape.len() == 2 {
            (shape[0] as usize, shape[1] as usize)
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

        let codes_cpu = codes_t
            .flatten(0, -1)
            .to_kind(Kind::Int64)
            .to_device(Device::Cpu);
        let n = codes_cpu.numel();
        let mut flat = vec![0i64; n as usize];
        codes_cpu.copy_data(&mut flat, n);

        Array2::from_shape_vec((rows, cols), flat).context("Failed to reshape encoded VQ codes")
    }
}
