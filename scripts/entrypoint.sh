#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${ARKTTS_MODEL_DIR:-/app/model}"
VOICES_DIR="${ARKTTS_VOICES_DIR:-/app/voices}"

# ---------------------------------------------------------------------------
# 1. Auto-detect NVIDIA GPU vs CPU access at container startup
# ---------------------------------------------------------------------------
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=1
elif [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ]; then
    HAS_GPU=1
fi

if [ "${HAS_GPU}" -eq 1 ]; then
    DRIVER_MAJ=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)
    if [ -n "${DRIVER_MAJ}" ] && [ "${DRIVER_MAJ}" -lt 550 ] 2>/dev/null; then
        echo "❌ Host NVIDIA driver ${DRIVER_MAJ}.x is too old for CUDA 12.4 LibTorch (requires >= 550.x)."
        exit 1
    fi
    RAW_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d " ." || true)
    DETECTED_MODE="GPU (NVIDIA LibTorch CUDA sm_${RAW_CAP:-unknown} - FP16)"
    export ARKTTS_DEVICE="${ARKTTS_DEVICE:-gpu}"
    export ARKTTS_PRECISION="${ARKTTS_PRECISION:-fp16}"
    export ARKTTS_WORKERS="${ARKTTS_WORKERS:-1}"
    export LD_PRELOAD="/opt/libtorch/lib/libtorch_global_deps.so /opt/libtorch/lib/libtorch_cuda.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
else
    DETECTED_MODE="CPU (LibTorch Multi-Thread - FP32)"
    export ARKTTS_DEVICE="${ARKTTS_DEVICE:-cpu}"
    export ARKTTS_PRECISION="${ARKTTS_PRECISION:-fp32}"
    export ARKTTS_CODEC_PRECISION="${ARKTTS_CODEC_PRECISION:-fp32}"
    export ARKTTS_WORKERS="${ARKTTS_WORKERS:-1}"
    export OMP_NUM_THREADS="${ARKTTS_THREADS:-4}"
    export MKL_NUM_THREADS="${ARKTTS_THREADS:-4}"
fi

export LIBTORCH="${LIBTORCH:-/opt/libtorch}"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo "🚀 Audio8-TTS-RS Gateway Starting (Native LibTorch / tch-rs)"
echo "   Hardware Mode     : ${DETECTED_MODE}"
echo "   ARKTTS_DEVICE     : ${ARKTTS_DEVICE}"
echo "   ARKTTS_PRECISION  : ${ARKTTS_PRECISION}"
echo "   MODEL_DIR         : ${MODEL_DIR}"
echo "   VOICES_DIR        : ${VOICES_DIR}"
echo "============================================================"

if [ ! -f "${MODEL_DIR}/runtime_manifest.json" ]; then
    echo "❌ runtime_manifest.json missing from ${MODEL_DIR} — rebuild image with 'docker compose build'."
    exit 1
fi

echo "⚡ Launching binary: /app/audio8-tts-server on port ${PORT:-18473}..."
exec /app/audio8-tts-server
