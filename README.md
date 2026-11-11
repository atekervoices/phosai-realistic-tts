# Audio8-TTS-RS (`phosai/realistic-air-v1`)

Low-latency Rust TTS backend for **`phosai/realistic-air-v1`** (fine-tuned from `Edge0/Audio8-TTS-Realistic-Air` 0.6B DualAR), maintaining **100% API compatibility** with `spark-tts-rs` and LiveKit `phos-agent` (`sparktts.py`) while running on **ONNX Runtime (`ort`) across both GPU (CUDA / TensorRT / DirectML) and CPU**, with optional **SGLang-Omni** continuous-batching GPU proxy support.

---

## Architecture & Runtime Advice: ONNX Runtime (`ort`) vs `tch-rs` vs SGLang-Omni

| Dimension | **ONNX Runtime (`ort` crate in Rust)** *(Default)* | **`tch-rs` (LibTorch C++)** | **SGLang-Omni + Rust Gateway** (`--sglang-url`) |
| :--- | :--- | :--- | :--- |
| **Deployment Footprint** | Single self-contained Rust binary; no Python or PyTorch runtime needed at inference time. | Requires matching `libtorch` C++ ABI shared libraries (~2.5 GB) and manual TorchScript tracing of both SlowAR and FastAR. | Requires running Python + SGLang + CUDA Graphs alongside the Rust gateway (similar to how `vllm` ran alongside `spark-tts-rs`). |
| **GPU & CPU Portability** | **Runs on both GPU (`CUDA`, `TensorRT`, `DirectML`) and CPU (`INT8` / `FP32`)** via `ExecutionDevice::Auto` with automatic fallback. | Tied to a specific LibTorch CPU or CUDA build at compile time. | NVIDIA GPU only (requires FlashAttention / FlashInfer). |
| **Single-Stream Latency (TTFB)** | **~180–320 ms TTFB on GPU**, **~450–700 ms on CPU** using 8-frame initial streaming window (`ProgressiveAudio8Streamer`) + pre-encoded `codes.npy` speaker profiles. | Comparable to ONNX on GPU, slower on CPU without INT8 dynamic quantization. | **~120 ms TTFB (RTF 0.116)** under high concurrent load via CUDA Graphs and RadixAttention. |
| **Recommendation** | **Best default for single-binary GPU & CPU deployment** (edge servers, single-GPU VMs, CPU fallback). | Not needed—`ort` covers both SlowAR/FastAR and `ArkttsCodec` in one unified runtime. | **Supported via `--sglang-url`** when scaling to many concurrent streams on a dedicated multi-user GPU server. |

---

## Key Low-Latency Ideas Borrowed from `Audio8_TTS/onnx_runtime` & `spark-tts-rs`

1. **Zero Reference-Encoding Latency (`voices/*/codes.npy`)**:
   - All **94 speaker references** from `speaker_references_bundle.zip` (`speaker_registry.json`) are pre-encoded into `[10, T]` VQ codec matrices (`codes.npy` + `meta.json`) during `scripts/export_onnx.py` (or lazily encoded once on first request and cached in memory + disk).
   - Dynamic voice cloning (`/v1/audio/speech/clone/upload` & `/v1/audio/speech/clone/ws`) caches encoded `[10, T]` reference codes in memory keyed by **SHA-256 hash** of the reference audio bytes.
2. **Sliding-Window Streaming Vocoder with Guard Frames & Crossfade (`ProgressiveAudio8Streamer`)**:
   - Emits the **first audio chunk after 8 frames (~370 ms)** for fast Time-To-First-Audio (TTFA) and subsequent chunks every **12 frames (~550 ms)**.
   - Decodes a bounded sliding window of `stream_context_frames = 128` frames and trims `stream_guard_frames = 1` (`2048` samples) from the right edge of non-final chunks so conv-transpose right-boundary transients never produce clicks.
   - Applies a **5 ms fade-in** on the first chunk, a **10 ms linear crossfade** between consecutive chunks, and a **40 Hz high-pass filter** to eliminate DC offset.
3. **Repetition Aware Sampling (RAS) + First-Frame EOS Masking + Loop Detector**:
   - Masks `<|im_end|>` on step 0 so the model never terminates with 0 frames.
   - Tracks the last 10 semantic tokens (`ras_window_size = 10`) and resamples with `temperature = 1.0, top_p = 0.9` on repetition.
   - Includes `spark-tts-rs`'s cyclical tail-pattern detector (`check_semantic_repetition`, periods `1..=12`) to prevent runaway token loops.
4. **Drop-in Sample Rate Adaptation (`16000 Hz` & `44100 Hz`)**:
   - Native Audio8 codec outputs `44100 Hz`, while existing LiveKit `sparktts.py` agents initialize at `16000 Hz`.
   - `OUTPUT_SAMPLE_RATE` defaults to `16000` (configurable to `44100` via `--output-sample-rate 44100` or per-request `"sample_rate": 44100`) using a Lanczos-3 windowed-sinc anti-aliased resampler.

---

## Quick Start

### 1. Export `phosai/realistic-air-v1` to ONNX & Pre-encode All 94 Speakers
```bash
export HF_TOKEN="your_huggingface_token_here"

# Export FP32 ONNX (recommended for GPU CUDAExecutionProvider & CPU)
python scripts/export_onnx.py \
  --model-id phosai/realistic-air-v1 \
  --hf-token "$HF_TOKEN" \
  --output-dir model \
  --voices-dir voices \
  --registry speaker_registry.json \
  --precision fp32

# Or export INT8 quantized ONNX for fast CPU inference:
python scripts/export_onnx.py \
  --model-id phosai/realistic-air-v1 \
  --precision int8
```

### 2. Run the Rust Server (GPU with automatic CPU fallback)
```bash
cargo run --release --features download-binaries --bin server -- \
  --model-dir model \
  --voices-dir voices \
  --device auto \
  --output-sample-rate 16000 \
  --port 18473
```

---

## API Endpoints (100% Compatible with `spark-tts-rs`)

- **`GET /docs`** — Interactive Swagger UI documentation
- **`POST /v1/audio/speech/stream`** — HTTP chunked `audio/pcm` (`s16le`) streaming (`AudioRequest`)
- **`GET /v1/audio/speech/stream/ws`** — Real-time WebSocket streaming (`AudioRequest` + `"abort"`/`"interrupt"` cancellation)
- **`POST /v1/audio/speech/clone/upload`** — Multipart form voice cloning (`text`, `reference_audio`, optional `reference_text`, `language`, `temperature`)
- **`GET /v1/audio/speech/clone/ws`** — WebSocket voice cloning (`CloneWsRequest`)
- **`POST /v1/audio/speech`** & **`POST /api/tts`** — OpenAI-compatible WAV/PCM synthesis
- **`GET /api/voices`** — Lists all 94 bundled speaker references and registered custom voices
- **`POST /api/voices/register`** — Register a new persistent voice from uploaded audio + transcript
- **`GET /api/health`** — Active execution provider (`CUDAExecutionProvider` / `DirectMLExecutionProvider` / `CPUExecutionProvider`) & model status

