# Multi-stage Dockerfile for Audio8-TTS-RS (Native Rust + LibTorch / TorchScript GPU & CPU)
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Rust + LibTorch 2.5.1 (CUDA 12.4) Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::ForceIPv4 "true";' >> /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.5.1 + CUDA 12.4 libraries for torch-sys / tch 0.18 linking
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/cu124

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_HTTP_TIMEOUT=120
ENV CARGO_NET_RETRY=10

WORKDIR /app

# 1. Pre-build and cache Rust + torch-sys dependencies before copying src/
COPY phosai-normalizer-rs ./phosai-normalizer-rs
COPY Cargo.toml ./
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    mkdir -p src examples && \
    echo "" > src/lib.rs && \
    echo "fn main() {}" > examples/server.rs && \
    cargo build --release --bin server && \
    rm -rf src examples

# 2. Copy application source and compile audio8-tts-rs
COPY src ./src
COPY examples ./examples
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    touch src/lib.rs examples/server.rs && \
    cargo build --release --bin server && \
    strip --strip-unneeded /app/target/release/server 2>/dev/null || true

# Collect LibTorch + CUDA 12.4 / cuDNN shared libraries cleanly for the slim runner image
RUN mkdir -p /opt/libtorch/lib && \
    TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))") && \
    cp -a "${TORCH_LIB}/." /opt/libtorch/lib/ && \
    NVIDIA_LIB=$(python3 -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true) && \
    if [ -n "${NVIDIA_LIB}" ]; then find "${NVIDIA_LIB}" -name '*.so*' -exec cp -a {} /opt/libtorch/lib/ \; 2>/dev/null || true; fi && \
    cp -a /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcupti.so* /opt/libtorch/lib/ 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: TorchScript Exporter (PyTorch 2.5.1) + Pre-Encoded 94 Speaker Voices
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS model-exporter

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
        "transformers>=4.57.0,<5" \
        "huggingface_hub>=0.30.0,<1.0" \
        soundfile \
        scipy \
        safetensors \
        tqdm \
        numpy

WORKDIR /app
COPY speaker_registry.json /app/speaker_registry.json
COPY speaker_references /app/speaker_references
COPY scripts/export_torchscript.py /app/scripts/export_torchscript.py

ARG HF_TOKEN=""
ARG MODEL_ID=phosai/realistic-v1
ARG EXPORT_PRECISION=fp16,fp32

RUN --mount=type=cache,target=/root/.cache/huggingface \
    python3 -u /app/scripts/export_torchscript.py \
        --model-id "${MODEL_ID}" \
        --hf-token "${HF_TOKEN}" \
        --output-dir /app/model \
        --voices-dir /app/voices \
        --registry /app/speaker_registry.json \
        --precision "${EXPORT_PRECISION}"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Self-Contained CUDA 12 / LibTorch Runtime Image
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::ForceIPv4 "true";' >> /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3 \
    libsndfile1 \
    libgomp1 \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/libtorch/lib /opt/libtorch/lib
RUN ldconfig /opt/libtorch/lib

COPY --from=builder /app/target/release/server /app/audio8-tts-server
COPY --from=model-exporter /app/model /app/model
COPY --from=model-exporter /app/voices /app/voices
COPY speaker_registry.json /app/speaker_registry.json
COPY speaker_references /app/speaker_references
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN sed -i 's/\r$//' /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh

ENV LIBTORCH="/opt/libtorch"
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV ARKTTS_MODEL_DIR="/app/model"
ENV ARKTTS_VOICES_DIR="/app/voices"
ENV ARKTTS_DEVICE="auto"
ENV OUTPUT_SAMPLE_RATE="16000"
ENV PORT="18473"

EXPOSE 18473

ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
