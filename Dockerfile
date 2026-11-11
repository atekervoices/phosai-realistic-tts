# Multi-stage Dockerfile for Audio8-TTS-RS (ONNX Runtime GPU + CPU)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    python3 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# 1. Pre-build and cache all 321 Rust dependencies before copying src/
COPY phosai-normalizer-rs ./phosai-normalizer-rs
COPY Cargo.toml ./
RUN mkdir -p src examples && \
    echo "" > src/lib.rs && \
    echo "fn main() {}" > examples/server.rs && \
    cargo build --release --features download-binaries --bin server && \
    rm -rf src examples

# 2. Copy actual application source and compile only audio8-tts-rs
COPY scripts/ort_compat.cpp ./ort_compat.cpp
COPY build.rs ./build.rs
COPY src ./src
COPY examples ./examples
RUN g++ -O2 -fPIC -c ort_compat.cpp -o ort_compat.o && \
    ar rcs /usr/lib/x86_64-linux-gnu/libort_compat.a ort_compat.o && \
    ar rcs /app/libort_compat.a ort_compat.o && \
    touch src/lib.rs examples/server.rs && \
    cargo build --release --features download-binaries --bin server && \
    mkdir -p /app/runtime_libs && \
    (find /app/target/release -maxdepth 4 -name "*.so*" -exec cp -f {} /app/runtime_libs/ \; 2>/dev/null || true) && \
    (find /root/.cache/ort.pyke.io -name "*.so*" -exec cp -f {} /app/runtime_libs/ \; 2>/dev/null || true)

# =============================================================================
# Stage 2: Pre-Bake ONNX Models (fp32 + int8) & 94 Speaker Voices
# =============================================================================
FROM python:3.11-slim AS model-exporter

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
        "transformers>=4.57.0,<5" \
        "huggingface_hub>=0.30.0,<1.0" \
        onnx \
        onnxscript \
        onnxruntime \
        soundfile \
        scipy \
        safetensors \
        tqdm \
        numpy

WORKDIR /app
COPY speaker_registry.json /app/speaker_registry.json
COPY speaker_references /app/speaker_references
COPY scripts/export_onnx.py /app/scripts/export_onnx.py

ARG HF_TOKEN=""
ARG MODEL_ID=phosai/realistic-v1

# Exports fp32 (GPU/CPU) + int8 fast AR ONNX models and all 94 speaker voices
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python3 -u /app/scripts/export_onnx.py \
        --model-id "${MODEL_ID}" \
        --hf-token "${HF_TOKEN}" \
        --output-dir /app/model \
        --voices-dir /app/voices \
        --registry /app/speaker_registry.json \
        --precision fp32

# =============================================================================
# Stage 3: Self-Contained Runtime Image (Pre-Baked Models + Voices)
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/server /app/audio8-tts-server
COPY --from=builder /app/runtime_libs/ /app/
COPY --from=model-exporter /app/model /app/model
COPY --from=model-exporter /app/voices /app/voices
COPY speaker_registry.json /app/speaker_registry.json
COPY speaker_references /app/speaker_references
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN sed -i 's/\r$//' /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh

ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"
ENV ARKTTS_MODEL_DIR="/app/model"
ENV ARKTTS_VOICES_DIR="/app/voices"
ENV ARKTTS_DEVICE="auto"
ENV OUTPUT_SAMPLE_RATE="16000"
ENV PORT="18473"

EXPOSE 18473

ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
