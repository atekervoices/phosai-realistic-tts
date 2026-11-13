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
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# Remove mismatched flashinfer-jit-cache package and any leftover namespace directory so FlashInfer falls back cleanly to AOT
"${PYTHON_BIN}" -m pip uninstall -y flashinfer-jit-cache >/dev/null 2>&1 || true
"${PYTHON_BIN}" -c "
import pathlib, shutil, sys
for p in sys.path:
    d = pathlib.Path(p)
    if d.is_dir():
        for m in d.glob('flashinfer_jit_cache*'):
            if m.is_dir():
                shutil.rmtree(m, ignore_errors=True)
            else:
                m.unlink(missing_ok=True)
" || true

# Auto-download Hugging Face snapshot if not already present on disk
TARGET_DIR="/models/$(echo "${MODEL}" | tr '/' '_')"

if [[ -f "${MODEL}/config.json" ]]; then
  echo "✅ Found model config at ${MODEL}/config.json"
elif [[ -f "${TARGET_DIR}/config.json" ]]; then
  echo "✅ Found cached model snapshot in ${TARGET_DIR}"
  MODEL="${TARGET_DIR}"
else
  echo "📥 Snapshot '${MODEL}' not found locally. Downloading from HuggingFace..."
  mkdir -p "${TARGET_DIR}"
  "${PYTHON_BIN}" -c "
import os, sys
from huggingface_hub import snapshot_download

model_id = '${MODEL}'
target_dir = '${TARGET_DIR}'
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or None
is_authed = bool(token and token.strip())
print(f'Starting snapshot download for {model_id} (authenticated: {is_authed})...')

try:
    snapshot_download(repo_id=model_id, local_dir=target_dir, token=token if is_authed else None)
    print(f'✅ Successfully downloaded {model_id} to {target_dir}')
except Exception as e:
    print(f'❌ Download failed for {model_id}: {e}', file=sys.stderr)
    if '401' in str(e) or 'Unauthorized' in str(e):
        print('\n[CRITICAL] 401 Unauthorized: The repository is private or gated.', file=sys.stderr)
        print('Please supply your Hugging Face token via one of:', file=sys.stderr)
        print('  1. Add HF_TOKEN=hf_... to ~/phosai-realistic-tts/.env', file=sys.stderr)
        print('  2. export HF_TOKEN=hf_... in your shell before running docker compose up', file=sys.stderr)
        print('  3. Or run huggingface-cli login on the host with ~/.cache/huggingface mounted.\n', file=sys.stderr)
    sys.exit(1)
"
  MODEL="${TARGET_DIR}"
fi

exec "${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
  --model-path "${MODEL}" \
  --config "${CONFIG}" \
  --model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}"
