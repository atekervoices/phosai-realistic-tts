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

/// Prepare 44.1 kHz mono reference audio for `codec_encoder_fp32.pt`:
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

const LANCZOS_LOBES: f64 = 3.0;

/// Stateful streaming Lanczos-3 resampler preserving filter support and fractional
/// phase across chunk boundaries so chunked streaming output matches one-shot [`resample`].
#[derive(Debug, Clone)]
pub struct StreamingLanczosResampler {
    src_rate: u32,
    dst_rate: u32,
    ratio: f64,
    filter_scale: f64,
    support: f64,
    buf: Vec<f32>,
    buf_start: isize,
    next_out_idx: usize,
    total_in_samples: usize,
    first_sample: Option<f32>,
    last_sample: Option<f32>,
}

impl StreamingLanczosResampler {
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        let active = src_rate > 0 && dst_rate > 0 && src_rate != dst_rate;
        let ratio = if active {
            src_rate as f64 / dst_rate as f64
        } else {
            1.0
        };
        // For downsampling (e.g. 44100 -> 16000), widen the sinc kernel support by `dst/src` for anti-aliasing
        let filter_scale = if active && dst_rate < src_rate {
            dst_rate as f64 / src_rate as f64
        } else {
            1.0
        };
        Self {
            src_rate,
            dst_rate,
            ratio,
            filter_scale,
            support: LANCZOS_LOBES / filter_scale,
            buf: Vec::new(),
            buf_start: 0,
            next_out_idx: 0,
            total_in_samples: 0,
            first_sample: None,
            last_sample: None,
        }
    }

    pub fn push(&mut self, chunk: &[f32], is_final: bool) -> Vec<f32> {
        if self.src_rate == 0 || self.dst_rate == 0 || self.src_rate == self.dst_rate {
            return chunk.to_vec();
        }
        if !chunk.is_empty() {
            if self.first_sample.is_none() {
                self.first_sample = Some(chunk[0]);
            }
            self.last_sample = Some(chunk[chunk.len() - 1]);
            self.buf.extend_from_slice(chunk);
            self.total_in_samples += chunk.len();
        }
        if self.total_in_samples == 0 {
            return Vec::new();
        }

        let buf_end = self.total_in_samples as isize;
        let target_total_out = if is_final {
            ((self.total_in_samples as f64) / self.ratio).round() as usize
        } else {
            usize::MAX
        };
        let first_val = self.first_sample.unwrap_or(0.0);
        let last_val = self.last_sample.unwrap_or(0.0);
        let mut out = Vec::new();

        while self.next_out_idx < target_total_out {
            let center = (self.next_out_idx as f64 + 0.5) * self.ratio - 0.5;
            let left = (center - self.support).floor() as isize;
            let right = (center + self.support).ceil() as isize;

            if !is_final && right >= buf_end {
                break;
            }

            let mut sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;

            for j in left..=right {
                let sample = if j < 0 {
                    first_val
                } else if j >= buf_end {
                    last_val
                } else {
                    self.buf[(j - self.buf_start) as usize]
                };
                let x = (center - j as f64) * self.filter_scale;
                let w = lanczos_kernel(x, LANCZOS_LOBES);
                sum += (sample as f64) * w;
                weight_sum += w;
            }

            let val = if weight_sum.abs() > 1e-12 {
                (sum / weight_sum) as f32
            } else {
                0.0
            };
            out.push(val.clamp(-1.0, 1.0));
            self.next_out_idx += 1;
        }

        if is_final {
            self.buf.clear();
            self.buf_start = buf_end;
        } else {
            let next_center = (self.next_out_idx as f64 + 0.5) * self.ratio - 0.5;
            let next_left = (next_center - self.support).floor() as isize;
            let keep_from = next_left.max(self.buf_start).min(buf_end);
            if keep_from > self.buf_start {
                let drop_n = (keep_from - self.buf_start) as usize;
                self.buf.drain(..drop_n);
                self.buf_start = keep_from;
            }
        }

        out
    }

    pub fn finish(&mut self) -> Vec<f32> {
        self.push(&[], true)
    }
}

/// High-quality windowed-sinc (Lanczos-3) resampler with anti-aliasing filter for downsampling.
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if samples.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return samples.to_vec();
    }
    StreamingLanczosResampler::new(src_rate, dst_rate).push(samples, true)
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

/// Convert 16-bit signed little-endian PCM bytes (`s16le`) to `f32` samples in `[-1.0, 1.0]`.
pub fn pcm16le_bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    let num_samples = bytes.len() / 2;
    let mut samples = Vec::with_capacity(num_samples);
    for chunk in bytes.chunks_exact(2) {
        let pcm = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(pcm as f32 / 32768.0);
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(sample_rate: u32, freq: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.7 * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn streaming_lanczos_matches_oneshot_downsample() {
        let signal = sine_wave(44100, 440.0, 22531);
        let expected = resample(&signal, 44100, 16000);

        let chunk_sizes = [1607usize, 5703, 10240, 1, 0, 441, 4096];
        let mut resampler = StreamingLanczosResampler::new(44100, 16000);
        let mut streamed = Vec::new();
        let mut pos = 0;
        let mut idx = 0;
        while pos < signal.len() {
            let sz = chunk_sizes[idx % chunk_sizes.len()].min(signal.len() - pos);
            streamed.extend(resampler.push(&signal[pos..pos + sz], false));
            pos += sz;
            idx += 1;
        }
        streamed.extend(resampler.finish());

        assert_eq!(streamed.len(), expected.len());
        let max_diff = streamed
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-6, "max_diff={max_diff}");
    }

    #[test]
    fn streaming_lanczos_matches_oneshot_upsample_and_identity() {
        let signal = sine_wave(16000, 310.0, 4096);
        let expected_up = resample(&signal, 16000, 44100);
        let mut up = StreamingLanczosResampler::new(16000, 44100);
        let mut streamed_up = Vec::new();
        for piece in signal.chunks(333) {
            streamed_up.extend(up.push(piece, false));
        }
        streamed_up.extend(up.finish());
        assert_eq!(streamed_up, expected_up);

        let mut id = StreamingLanczosResampler::new(44100, 44100);
        let mut streamed_id = id.push(&signal[..1000], false);
        streamed_id.extend(id.push(&signal[1000..], true));
        assert_eq!(streamed_id, signal);
    }
}


