#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export MODEL_ID="${MODEL_ID:-phosai/realistic-air-v1}"
export ARKTTS_DEVICE="${ARKTTS_DEVICE:-auto}"
export OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-16000}"

echo "=== Step 1: Exporting ${MODEL_ID} to ONNX & Pre-encoding 94 Speaker Voices ==="
if [ ! -f "model/runtime_manifest.json" ]; then
    python3 scripts/export_onnx.py \
        --model-id "${MODEL_ID}" \
        --hf-token "${HF_TOKEN}" \
        --output-dir "model" \
        --voices-dir "voices" \
        --registry "speaker_registry.json" \
        --precision "fp32"
else
    echo "ONNX model already exported in ./model/runtime_manifest.json"
fi

echo "=== Step 2: Building & Starting Audio8-TTS-RS Server ==="
cargo run --release --features download-binaries --bin server -- \
    --model-dir model \
    --voices-dir voices \
    --device "${ARKTTS_DEVICE}" \
    --output-sample-rate "${OUTPUT_SAMPLE_RATE}" \
    --port "${PORT:-18473}"

