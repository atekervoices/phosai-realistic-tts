#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${ARKTTS_MODEL_DIR:-/app/model}"
VOICES_DIR="${ARKTTS_VOICES_DIR:-/app/voices}"

# ---------------------------------------------------------------------------
# 1. Auto-detect GPU vs CPU access at container startup
# ---------------------------------------------------------------------------
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=1
elif [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ]; then
    HAS_GPU=1
fi

if [ "${HAS_GPU}" -eq 1 ]; then
    DETECTED_MODE="GPU (NVIDIA CUDA / TensorRT - FP32)"
    export ARKTTS_DEVICE="${ARKTTS_DEVICE:-gpu}"
    export ARKTTS_PRECISION="${ARKTTS_PRECISION:-fp32}"
    export ARKTTS_WORKERS="${ARKTTS_WORKERS:-1}"
else
    DETECTED_MODE="CPU (ONNX Runtime Multi-Thread - INT8/FP32)"
    export ARKTTS_DEVICE="${ARKTTS_DEVICE:-cpu}"
    export ARKTTS_PRECISION="${ARKTTS_PRECISION:-fp32}"
    export ARKTTS_WORKERS="${ARKTTS_WORKERS:-1}"
fi

# Ensure compatibility symlinks exist in MODEL_DIR
if [ -d "${MODEL_DIR}" ]; then
    [ -f "${MODEL_DIR}/slow_ar_fp32.onnx" ] && [ ! -f "${MODEL_DIR}/slow_ar.onnx" ] && ln -sf slow_ar_fp32.onnx "${MODEL_DIR}/slow_ar.onnx" || true
    [ -f "${MODEL_DIR}/slow_ar_fp32.onnx" ] && [ ! -f "${MODEL_DIR}/slow_ar_int8.onnx" ] && ln -sf slow_ar_fp32.onnx "${MODEL_DIR}/slow_ar_int8.onnx" || true
    [ -f "${MODEL_DIR}/fast_ar_fp32.onnx" ] && [ ! -f "${MODEL_DIR}/fast_ar.onnx" ] && ln -sf fast_ar_fp32.onnx "${MODEL_DIR}/fast_ar.onnx" || true
    [ -f "${MODEL_DIR}/codec_decoder_fp32.onnx" ] && [ ! -f "${MODEL_DIR}/codec_decoder.onnx" ] && ln -sf codec_decoder_fp32.onnx "${MODEL_DIR}/codec_decoder.onnx" || true
fi

# ---------------------------------------------------------------------------
# 2. Auto-detect optional SGLang-Omni sidecar if reachable
# ---------------------------------------------------------------------------
if [ -z "${SGLANG_URL:-}" ]; then
    if curl -sf --max-time 1 "http://sglang-audio8:18472/health" >/dev/null 2>&1; then
        export SGLANG_URL="http://sglang-audio8:18472"
    fi
fi

echo "============================================================"
echo "🚀 Audio8-TTS-RS Gateway Starting (Pre-Baked Image)"
echo "   Hardware Mode    : ${DETECTED_MODE}"
echo "   ARKTTS_DEVICE    : ${ARKTTS_DEVICE}"
echo "   ARKTTS_PRECISION : ${ARKTTS_PRECISION}"
echo "   MODEL_DIR        : ${MODEL_DIR}"
echo "   VOICES_DIR       : ${VOICES_DIR}"
echo "   SGLANG_URL       : ${SGLANG_URL:-<standalone Rust ONNX worker pool>}"
echo "============================================================"

if [ ! -f "${MODEL_DIR}/runtime_manifest.json" ]; then
    echo "❌ runtime_manifest.json missing from ${MODEL_DIR} — rebuild image with 'docker compose build'."
    exit 1
fi

echo "⚡ Launching binary: /app/audio8-tts-server on port ${PORT:-18473}..."
exec /app/audio8-tts-server
