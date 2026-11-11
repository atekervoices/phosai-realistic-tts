use anyhow::{Context, Result};
use std::io::Cursor;
use std::path::Path;

/// Load a WAV file from disk, downmix to mono `f32` in `[-1.0, 1.0]`, and resample to `target_sample_rate`.
pub fn load_wav<P: AsRef<Path>>(path: P, target_sample_rate: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path.as_ref())
        .with_context(|| format!("Failed to open WAV file: {}", path.as_ref().display()))?;
    decode_wav_reader(reader, target_sample_rate)
}

/// Load a WAV file from raw in-memory bytes, downmix to mono `f32`, and resample to `target_sample_rate`.
pub fn load_wav_from_bytes(bytes: &[u8], target_sample_rate: u32) -> Result<Vec<f32>> {
    let cursor = Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor).context("Failed to parse WAV bytes")?;
    decode_wav_reader(reader, target_sample_rate)
}

fn decode_wav_reader<R: std::io::Read>(
    mut reader: hound::WavReader<R>,
    target_sample_rate: u32,
) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f32 / max_val).clamp(-1.0, 1.0)))
                .collect::<std::result::Result<Vec<f32>, _>>()?
        }
    };

    // Downmix multi-channel audio to mono
    let mono: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
            .collect()
    } else {
        raw_samples
    };

    if spec.sample_rate == target_sample_rate || target_sample_rate == 0 {
        Ok(mono)
    } else {
        Ok(resample(&mono, spec.sample_rate, target_sample_rate))
    }
}

/// Prepare 44.1 kHz mono reference audio for `codec_encoder.onnx` (matching `VoiceRegistration._prepare_audio`
/// in `Audio8_TTS/onnx_runtime/arktts_runtime/registration.py`):
/// - Clamps to `[-1.0, 1.0]`
/// - Ensures length is a non-zero multiple of `hop_length` (2048 samples) by zero-padding the tail.
pub fn prepare_reference_audio(samples: &[f32], hop_length: usize) -> Result<Vec<f32>> {
    if samples.is_empty() {
        anyhow::bail!("Reference audio cannot be empty");
    }
    let hop = hop_length.max(1);
    let remainder = samples.len() % hop;
    let pad = if remainder == 0 { 0 } else { hop - remainder };
    let mut out = Vec::with_capacity(samples.len() + pad);
    for &s in samples {
        let v = if s.is_finite() {
            s.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        out.push(v);
    }
    if pad > 0 {
        out.resize(samples.len() + pad, 0.0);
    }
    Ok(out)
}

/// High-quality windowed-sinc (Lanczos-3) resampler with anti-aliasing filter for downsampling.
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if samples.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }

    // For downsampling (e.g. 44100 -> 16000), widen the sinc kernel support by `ratio` for anti-aliasing
    let filter_scale = if dst_rate < src_rate {
        dst_rate as f64 / src_rate as f64
    } else {
        1.0
    };
    let a = 3.0_f64; // Lanczos-3 lobes
    let support = a / filter_scale;

    let mut out = Vec::with_capacity(out_len);
    let n_in = samples.len() as isize;

    for i in 0..out_len {
        let center = (i as f64 + 0.5) * ratio - 0.5;
        let left = (center - support).floor() as isize;
        let right = (center + support).ceil() as isize;

        let mut sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for j in left..=right {
            let idx = j.clamp(0, n_in - 1) as usize;
            let x = (center - j as f64) * filter_scale;
            let w = lanczos_kernel(x, a);
            sum += (samples[idx] as f64) * w;
            weight_sum += w;
        }

        let val = if weight_sum.abs() > 1e-12 {
            (sum / weight_sum) as f32
        } else {
            0.0
        };
        out.push(val.clamp(-1.0, 1.0));
    }

    out
}

#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

#[inline]
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() >= a {
        0.0
    } else {
        sinc(x) * sinc(x / a)
    }
}

/// Save `f32` audio samples as a 16-bit PCM WAV file on disk.
pub fn save_wav<P: AsRef<Path>>(path: P, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let amplitude = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(amplitude)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Encode `f32` audio samples into an in-memory 16-bit PCM WAV byte buffer.
pub fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec)?;
        for &sample in samples {
            let amplitude = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;
    }
    Ok(buf.into_inner())
}

/// Convert `f32` audio samples `[-1.0, 1.0]` to 16-bit signed little-endian PCM bytes (`s16le`).
pub fn f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = if s.is_finite() {
            s.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let pcm = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    bytes
}

/// Apply a linear fade-in over the first `fade_samples` to eliminate start clicks.
pub fn apply_fade_in(samples: &mut [f32], fade_samples: usize) {
    let len = samples.len().min(fade_samples);
    if len == 0 {
        return;
    }
    for (i, sample) in samples.iter_mut().take(len).enumerate() {
        let gain = i as f32 / len as f32;
        *sample *= gain;
    }
}

/// Apply a linear fade-out over the last `fade_samples` to eliminate end clicks.
pub fn apply_fade_out(samples: &mut [f32], fade_samples: usize) {
    let total = samples.len();
    let len = total.min(fade_samples);
    if len == 0 {
        return;
    }
    for i in 0..len {
        let gain = (len - 1 - i) as f32 / len as f32;
        samples[total - len + i] *= gain;
    }
}

/// First-order high-pass filter to remove DC offset and sub-bass rumble (< `cutoff_hz`).
pub fn apply_highpass_filter(samples: &mut [f32], cutoff_hz: f32, sample_rate: u32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0 / sample_rate as f32;
    let alpha = rc / (rc + dt);

    let mut prev_x = samples[0];
    let mut prev_y = 0.0_f32;
    samples[0] = 0.0;

    for x in samples.iter_mut().skip(1) {
        let curr_x = *x;
        let y = alpha * (prev_y + curr_x - prev_x);
        prev_x = curr_x;
        prev_y = y;
        *x = y;
    }
}

/// Convenience alias for `load_wav`.
#[inline]
pub fn load_audio_mono_resampled<P: AsRef<Path>>(path: P, target_sample_rate: u32) -> Result<Vec<f32>> {
    load_wav(path, target_sample_rate)
}

/// Convenience alias for `load_wav_from_bytes`.
#[inline]
pub fn decode_wav_bytes_resampled(bytes: &[u8], target_sample_rate: u32) -> Result<Vec<f32>> {
    load_wav_from_bytes(bytes, target_sample_rate)
}

/// Convenience alias for `resample` (Lanczos-3 windowed sinc resampler).
#[inline]
pub fn resample_lanczos(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    resample(samples, src_rate, dst_rate)
}

/// Convenience alias for `encode_wav_bytes`.
#[inline]
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    encode_wav_bytes(samples, sample_rate)
}

/// Convenience alias for `f32_to_pcm16_bytes`.
#[inline]
pub fn samples_to_pcm16le_bytes(samples: &[f32]) -> Vec<u8> {
    f32_to_pcm16_bytes(samples)
}


