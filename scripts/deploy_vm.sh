#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export MODEL_ID="${MODEL_ID:-phosai/realistic-v1}"
export ARKTTS_DEVICE="${ARKTTS_DEVICE:-auto}"
export EXPORT_PRECISION="${EXPORT_PRECISION:-fp16,fp32}"
export OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-16000}"

echo "=== Step 1: Exporting ${MODEL_ID} to TorchScript & Pre-encoding 94 Speaker Voices ==="
if [ ! -f "model/runtime_manifest.json" ]; then
    python3 -u scripts/export_torchscript.py \
        --model-id "${MODEL_ID}" \
        --hf-token "${HF_TOKEN}" \
        --output-dir "model" \
        --voices-dir "voices" \
        --registry "speaker_registry.json" \
        --precision "${EXPORT_PRECISION}"
else
    echo "TorchScript model already exported in ./model/runtime_manifest.json"
fi

echo "=== Step 2: Building & Starting Audio8-TTS-RS LibTorch Server ==="
export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

cargo run --release --bin server -- \
    --model-dir model \
    --voices-dir voices \
    --device "${ARKTTS_DEVICE}" \
    --output-sample-rate "${OUTPUT_SAMPLE_RATE}" \
    --port "${PORT:-18473}"
