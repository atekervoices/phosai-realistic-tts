# Phosai Realistic TTS (`phosai/realistic-v1`)

High-performance, low-latency Rust TTS inference engine and microservice for **`phosai/realistic-v1`** (fine-tuned from `Edge0/Audio8-TTS-Realistic-Air` 0.6B DualAR).

The engine runs directly on **ONNX Runtime (`ort`)** across both **NVIDIA GPU** (CUDA / TensorRT) and **CPU** (multi-threaded INT8/FP32), featuring built-in African language text normalization, zero-latency reference encoding for 94 bundled voices, and 100% protocol compatibility with OpenAI TTS, `spark-tts-rs`, and LiveKit Agents (`/ws/tts`).

---

## Key Features

- **Blazing Fast Low Latency (TTFB < 200 ms on GPU, < 600 ms on CPU)**:
  - **Zero Reference-Encoding Latency**: All 94 speaker profiles from `speaker_registry.json` are pre-encoded into `[10, T]` VQ codec token matrices. No neural speech encoding is needed at request time.
  - **Sliding-Window Streaming Vocoder (`ProgressiveAudio8Streamer`)**: Emits the first audio chunk within 8 frames (~370 ms) and subsequent chunks every 12 frames (~550 ms) with 10 ms inter-chunk crossfading to eliminate boundary clicks.
  - **Repetition-Aware Sampling (RAS)** + First-Frame EOS Masking + Cyclical Loop Detector prevents hallucinations or silence.
- **Unified GPU & CPU Runtime**:
  - Automatically detects NVIDIA GPU (`CUDAExecutionProvider` / `TensorRTExecutionProvider`) or falls back to multi-threaded CPU (`CPUExecutionProvider`).
  - Supports INT8 dynamic quantization for CPU deployments and FP32 for GPU.
- **Multilingual Native Text Normalization (`phosai-normalizer-rs`)**:
  - Built-in rule-based text normalizer for **Luganda (`lug`)**, **Swahili (`swa`)**, **Acholi (`ach`)**, **Runyankole (`nyn`)**, **Ateso (`teo`)**, **Kinyarwanda (`kin`)**, and **English (`eng`)**.
  - Automatically handles currency symbols, cardinal & ordinal numbers, dates, times, measurements, abbreviations, and phone numbers.
- **Drop-in Protocol Compatibility**:
  - **OpenAI Audio API**: `POST /v1/audio/speech`
  - **Chunked PCM Streaming**: `POST /v1/audio/speech/stream`
  - **Bi-directional WebSockets**: `GET /v1/audio/speech/stream/ws`
  - **LiveKit Agent Protocol**: `GET /ws/tts` (drop-in replacement for LiveKit `sparktts.py` plugins)
  - **Zero-Shot Voice Cloning**: `POST /v1/audio/speech/clone/upload` and `GET /v1/audio/speech/clone/ws`
  - **Interactive Swagger UI**: `GET /docs` and `GET /openapi.json`
- **Configurable Output Sample Rate**:
  - Native model sample rate is `44100 Hz`.
  - Configurable to standard telephony/agent `16000 Hz` or native `44100 Hz` via anti-aliased Lanczos-3 resampler.

---

## 94 Pre-Baked African Speaker Voices

The engine bundles 94 pre-encoded speaker profiles with short aliases across diverse African accents and languages:

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

The easiest way to run the service is with Docker Compose. The Docker image pre-bakes the ONNX models and all 94 speaker voice matrices at build time.

### 1. Start the Container
```bash
docker compose up -d
```

### 2. View Server Logs
```bash
docker compose logs -f
```

The server listens on `http://0.0.0.0:18473`:
- **Swagger UI**: `http://localhost:18473/docs`
- **Health Check**: `http://localhost:18473/health`

### Configuration (`docker-compose.yml`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ARKTTS_DEVICE` | `auto` | Execution device: `auto` (GPU if available, else CPU), `cuda`, or `cpu`. |
| `ARKTTS_WORKERS` | `1` | Number of parallel ONNX Runtime session worker threads. |
| `ARKTTS_THREADS` | `4` | Number of intra-op compute threads per ONNX session. |
| `OUTPUT_SAMPLE_RATE` | `16000` | Target audio sample rate (`16000` or `44100`). |
| `PORT` | `18473` | Server HTTP and WebSocket port. |
| `HF_TOKEN` | *(optional)* | Hugging Face token (only required during container build if model is gated). |

---

## API Usage Examples

### 1. Synthesize Speech (OpenAI Compatible)
```bash
curl -X POST http://localhost:18473/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Habari za asubuhi, jina langu ni Musa. Karibu sana!",
    "voice": "crestai_waxal_tts:swa_male_3",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

### 2. Stream Audio via HTTP Chunked PCM
```bash
curl -X POST http://localhost:18473/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Oli otya nnyabo, leero tulina essanyu lingi okukusisinkana.",
    "voice": "crestai_vo_tts:lug_female_10"
  }' \
  --output stream.pcm
```

### 3. List Available Voices
```bash
curl http://localhost:18473/v1/voices
```

### 4. Check Health & Active Provider
```bash
curl http://localhost:18473/health
```
Response:
```json
{
  "status": "ok",
  "engine": "audio8-tts-rs",
  "provider": "CPUExecutionProvider",
  "workers": 1,
  "sample_rate": 16000,
  "native_sample_rate": 44100,
  "voices_loaded": 94
}
```

### 5. Dynamic Zero-Shot Voice Cloning
```bash
curl -X POST http://localhost:18473/v1/audio/speech/clone/upload \
  -F "reference_audio=@my_sample.wav" \
  -F "reference_text=This is the exact spoken text in the sample audio." \
  -F "text=Hello, this is my cloned voice speaking through Phosai Realistic TTS." \
  --output cloned_output.pcm
```

---

## Local Development & Manual Build

### Prerequisites
- **Rust 1.80+** (`cargo`, `rustc`)
- **Python 3.10+** (only for ONNX export script)
- **CMake & C++ Build Tools** (for compiling ONNX Runtime native bindings)

### Step 1: Export ONNX Models & Voices
```bash
python scripts/export_onnx.py \
  --model-id phosai/realistic-v1 \
  --output-dir model \
  --voices-dir voices \
  --registry speaker_registry.json \
  --precision fp32
```

### Step 2: Run the Server Binary
```bash
cargo run --release --features download-binaries --bin server -- \
  --model-dir model \
  --voices-dir voices \
  --device auto \
  --output-sample-rate 16000 \
  --port 18473
```

---

## LiveKit Agent Integration

To connect with LiveKit Python Agents using `sparktts.py`:
```python
from livekit.plugins import sparktts

# Point sparktts client to this service
tts = sparktts.TTS(
    base_url="http://localhost:18473",
    voice="crestai_vo_tts:lug_female_10",
    sample_rate=16000,
)
```

The service exposes the native WebSocket handler on `ws://localhost:18473/ws/tts`, streaming real-time PCM audio directly to LiveKit rooms.

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
