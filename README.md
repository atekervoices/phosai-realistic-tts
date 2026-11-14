# Phosai Realistic TTS (`phosai/realistic-v1`)

High-performance, low-latency native **Rust + LibTorch (`tch-rs` 0.18 / PyTorch 2.5.1)** TTS inference server for **`phosai/realistic-v1`** (fine-tuned from `Edge0/Audio8-TTS-Realistic-Air` 0.6B DualAR).

The engine executes pre-exported TorchScript `.pt` modules directly in-process via C++ LibTorch across both **NVIDIA GPU (`LibTorchCUDA`, FP16)** and **CPU (`LibTorchCPU`, FP32)**—with zero Python sidecars, zero ONNX Runtime, built-in African language text normalization, zero-latency reference encoding for 94 bundled voices, and protocol compatibility with OpenAI TTS, `spark-tts-rs`, and LiveKit Agents.

---

## Key Features

- **In-Process Native LibTorch (`tch-rs`) Pipeline**:
  - **GPU-Resident KV Cache**: Slow AR 0.6B transformer keeps its KV cache `[L, 1, H_kv, max_seq, head_dim]` pinned on CUDA VRAM and updates in-place without host-device transfers.
  - **Fused Single-Call `FastARLoop`**: Executes all 9 Fast AR codebook steps (`pos=0..9`) + GPU Top-K/Top-P/Temperature sampling in a single LibTorch call per frame.
  - **Zero-Latency Startup Warmup**: Pre-warms cuBLAS / cuDNN workspaces on startup (`cudnn.benchmark = 0`) so the very first client request responds without cold-start stalls.
  - **Sliding-Window Streaming Vocoder (`ProgressiveAudio8Streamer`)**: Emits the first audio chunk after 2 frames (~93 ms generated audio) and subsequent chunks every 3 frames with linear crossfading and guard trimming.
  - **Repetition-Aware Sampling (RAS)** + First-Frame EOS Masking + Cyclical Loop Detector prevents hallucinations or silence.
- **Zero Reference-Encoding Latency**: All 94 speaker profiles from `speaker_registry.json` are pre-encoded into `[10, T]` VQ codec token matrices at image build time.
- **Multilingual Native Text Normalization (`phosai-normalizer-rs`)**:
  - Built-in rule-based text normalizer for **Luganda (`lug`)**, **Swahili (`swa`)**, **Acholi (`ach`)**, **Runyankole (`nyn`)**, **Ateso (`teo`)**, **Kinyarwanda (`kin`)**, and **English (`eng`)**.
- **Drop-in Protocol Compatibility**:
  - **OpenAI Audio API**: `POST /v1/audio/speech`
  - **Chunked PCM Streaming**: `POST /v1/audio/speech/stream`
  - **Bi-directional WebSockets**: `GET /v1/audio/speech/stream/ws`
  - **Zero-Shot Voice Cloning**: `POST /v1/audio/speech/clone/upload` and `GET /v1/audio/speech/clone/ws`
  - **Interactive Swagger UI**: `GET /docs` and `GET /openapi.json`

---

## 94 Pre-Baked African Speaker Voices

| Language | Code | Bundled Voices / Sample Aliases |
| :--- | :--- | :--- |
| **Luganda** | `lug` | `crestai_vo_tts:lug_female_10` *(default)*, `crestai_vo_tts:lug_male_9`, `crestai_waxal_tts:lug_female_5`, `phosai_toby_lug_tts:lug_male_11`, `sunbird_salt_studio-lug:247` |
| **Swahili** | `swa` | `crestai_waxal_tts:swa_male_3`, `crestai_waxal_tts:swa_male_4`, `crestai_waxal_tts:swa_female_1`, `sunbird_salt_studio-swa:246` |
| **Acholi** | `ach` | `crestai_vo_tts:ach_female_9`, `crestai_vo_tts:ach_male_10`, `crestai_waxal_tts:ach_male_5`, `sunbird_salt_studio-ach:241` |
| **Runyankole** | `nyn` | `crestai_vo_tts:nyn_female_10`, `crestai_vo_tts:nyn_male_9`, `phosai_runyankole_nyn_tts:nyn_male_120`, `sunbird_salt_studio-nyn:243` |
| **Ateso / Karamojong** | `teo` / `kdj` | `crestai_vo_tts:teo_female_1`, `crestai_vo_tts:teo_male_1`, `phosai_atekervoice_kdj_tts_1:ATV0001`, `sunbird_salt_studio-teo:242` |
| **Kinyarwanda** | `kin` | `crestai_kin_tts:kin_female_1`, `phosai_afrispeak_kinyarwanda_kin_tts:kin_male_1` |
| **Nigerian & Cameroon Pidgin** | `pcm` / `wes` | `crestai_pidjin_tts:pcm_female_1` ... `pcm_male_5`, `phosai_nigerian_pidjin_cv_pcm_tts`, `phosai_cameroon_pidjin_cv_wes_tts` |
| **Wolof** | `wol` | `crestai_wolof_tts:wol_female_1`, `crestai_wolof_tts:wol_male_1` |
| **Twi & Fante** | `twi` / `fat` | `crestai_twi_akosua_female_speaker_tts:twi_female_5`, `crestai_waxal_tts:fat_female_2` |
| **Yoruba & Igbo** | `yor` / `ibo` | `crestai_waxal_tts:yor_female_3`, `crestai_waxal_tts:ibo_male_4` |
| **English & African French** | `eng` / `fr` | `crestai_vo_tts:eng_female_1`, `phosai_salt_eng_tts_1:eng_female_3`, `phosai_african_french_cv_tts:fr_male_algeria_1` |

---

## Quick Start with Docker (Recommended)

The single `audio8-tts-rs` container compiles `tch-rs` against LibTorch 2.5.1 (CUDA 12.4) and pre-bakes all TorchScript `.pt` modules plus all 94 speaker voice matrices at image build time.

```bash
cp .env.example .env
# Set HF_TOKEN in .env if phosai/realistic-v1 is private/gated
docker compose up -d --build
docker compose logs -f audio8-tts-rs
```

The server listens on `http://0.0.0.0:18473`:
- **Swagger UI**: `http://localhost:18473/docs`
- **Health Check**: `http://localhost:18473/health`

### Configuration (`docker-compose.yml`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ARKTTS_DEVICE` | `auto` | Execution device: `auto` (CUDA GPU if available, else CPU), `cuda`/`gpu`, or `cpu`. |
| `ARKTTS_PRECISION` | `fp16` | Slow/Fast AR TorchScript precision (`fp16` for GPU, `fp32` for CPU). |
| `ARKTTS_WORKERS` | `1` | Number of parallel LibTorch worker instances (`1` recommended on 8 GB VRAM GPUs). |
| `ARKTTS_THREADS` | `4` | CPU compute thread count when running on CPU. |
| `OUTPUT_SAMPLE_RATE` | `16000` | Target audio sample rate (`16000` or `44100`). |
| `PORT` | `18473` | Server HTTP and WebSocket port. |
| `HF_TOKEN` | *(optional)* | Hugging Face token (needed during Docker build if `phosai/realistic-v1` is private/gated). |

---

## Local Development & Manual Build

### Prerequisites
- **Rust 1.80+** (`cargo`, `rustc`)
- **Python 3.10+** with **PyTorch 2.5.1** (`cu124` for GPU or CPU wheel) and `LIBTORCH_USE_PYTORCH=1`

### Step 1: Export TorchScript Models & Voices
```bash
python scripts/export_torchscript.py \
  --model-id phosai/realistic-v1 \
  --output-dir model \
  --voices-dir voices \
  --registry speaker_registry.json \
  --precision fp16,fp32
```

### Step 2: Run the Server Binary
```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo run --release --bin server -- \
  --model-dir model \
  --voices-dir voices \
  --device auto \
  --output-sample-rate 16000 \
  --port 18473
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
