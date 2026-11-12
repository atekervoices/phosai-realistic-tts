#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/models/Audio8-TTS-Preview-0.6b}"
CONFIG="${CONFIG:-${BUNDLE_ROOT}/configs/audio8_tts_0_6b.yaml}"
MODEL_NAME="${MODEL_NAME:-audio8/tts-0.6b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

if [[ -n "${SGLANG_OMNI_ROOT:-}" ]]; then
  if [[ -d "${SGLANG_OMNI_ROOT}/sglang_omni" ]]; then
    export PYTHONPATH="${SGLANG_OMNI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  elif [[ -f "${SGLANG_OMNI_ROOT}/__init__.py" ]]; then
    PACKAGE_PARENT="$(cd "${SGLANG_OMNI_ROOT}/.." && pwd)"
    export PYTHONPATH="${PACKAGE_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
  else
    echo "Invalid SGLANG_OMNI_ROOT: ${SGLANG_OMNI_ROOT}" >&2
    exit 1
  fi
fi

if [[ -n "${SGLANG_OMNI_SITE_PACKAGES:-}" ]]; then
  if [[ ! -d "${SGLANG_OMNI_SITE_PACKAGES}" ]]; then
    echo "Invalid SGLANG_OMNI_SITE_PACKAGES: ${SGLANG_OMNI_SITE_PACKAGES}" >&2
    exit 1
  fi
  export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${SGLANG_OMNI_SITE_PACKAGES}"
fi

export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/audio8-flashinfer}"

# Auto-download Hugging Face snapshot if not already present on disk
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "📥 Snapshot '${MODEL}' not found locally. Downloading from HuggingFace..."
  TARGET_DIR="/models/$(echo "${MODEL}" | tr '/' '_')"
  mkdir -p "${TARGET_DIR}"
  "${PYTHON_BIN}" -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL}', local_dir='${TARGET_DIR}', token=os.environ.get('HF_TOKEN') or None)"
  MODEL="${TARGET_DIR}"
fi

exec "${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
  --model-path "${MODEL}" \
  --config "${CONFIG}" \
  --model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}"
