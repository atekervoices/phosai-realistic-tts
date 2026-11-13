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

# Patch sglang_omni/vendor/sglang/layers.py for older sglang/sgl_kernel in base image
"${PYTHON_BIN}" -c "
import pathlib
p = pathlib.Path('/opt/sglang-omni/sglang_omni/vendor/sglang/layers.py')
if p.is_file():
    txt = p.read_text()
    marker = '# patched-by-audio8-tts'
    if marker not in txt:
        start = txt.find('from sgl_kernel import top_k_top_p_sampling_from_probs')
        end = txt.find('# ---------------------------------------------------------------------------')
        if start != -1 and end != -1:
            replacement = '''# patched-by-audio8-tts
try:
    from sgl_kernel import top_k_top_p_sampling_from_probs
except ImportError:
    try:
        from flashinfer.sampling import top_k_top_p_sampling_from_probs
    except ImportError:
        top_k_top_p_sampling_from_probs = None
try:
    from sglang.srt.layers.activation import SiluAndMul
except ImportError:
    SiluAndMul = None
try:
    from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
except ImportError:
    LayerCommunicator = None
    LayerScatterModes = None
try:
    from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
except ImportError:
    get_attention_tp_rank = None
    get_attention_tp_size = None
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
try:
    from sglang.srt.layers.linear import ReplicatedLinear
except ImportError:
    ReplicatedLinear = None
try:
    from sglang.srt.layers.moe import (
        get_moe_a2a_backend,
        should_use_flashinfer_cutlass_moe_fp4_allgather,
    )
except ImportError:
    get_moe_a2a_backend = None
    should_use_flashinfer_cutlass_moe_fp4_allgather = None
try:
    from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
except ImportError:
    get_moe_impl_class = None
try:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
except ImportError:
    FusedMoE = None
try:
    from sglang.srt.layers.moe.topk import TopK
except ImportError:
    TopK = None
try:
    from sglang.srt.layers.moe.utils import RoutingMethodType
except ImportError:
    RoutingMethodType = None
try:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
except ImportError:
    QuantizationConfig = None
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
try:
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
except ImportError:
    MRotaryEmbedding = None
try:
    from sglang.srt.layers.utils import get_layer_id
except ImportError:
    get_layer_id = None
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

'''
            p.write_text(txt[:start] + replacement + txt[end:])
" || true

# Patch vendor/sglang/distributed.py and guard unrelated model registrations in model_runner.py
"${PYTHON_BIN}" -c "
import pathlib
d = pathlib.Path('/opt/sglang-omni/sglang_omni/vendor/sglang/distributed.py')
if d.is_file():
    txt = d.read_text()
    if '# patched-by-audio8-tts' not in txt and 'from sglang.srt.distributed import (' in txt:
        old = '''from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)'''
        new = '''# patched-by-audio8-tts
try:
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )
except ImportError:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_reduce,
        )
    except ImportError:
        def get_tensor_model_parallel_rank(): return 0
        def get_tensor_model_parallel_world_size(): return 1
        def tensor_model_parallel_all_reduce(x): return x'''
        if old in txt:
            d.write_text(txt.replace(old, new, 1))

mr = pathlib.Path('/opt/sglang-omni/sglang_omni/engines/ar/sglang_backend/model_runner.py')
if mr.is_file():
    txt = mr.read_text()
    old_super = '''        super().__init__(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )'''
    if old_super in txt:
        new_super = '''        import inspect
        _mr_params = inspect.signature(ModelRunner.__init__).parameters
        _mr_kwargs = {
            'model_config': model_config,
            'mem_fraction_static': server_args.mem_fraction_static,
            'gpu_id': gpu_id,
            'tp_rank': tp_rank,
            'tp_size': tp_size,
            'moe_ep_rank': moe_ep_rank,
            'moe_ep_size': moe_ep_size,
            'pp_rank': pp_rank,
            'pp_size': pp_size,
            'nccl_port': nccl_port,
            'server_args': server_args,
        }
        try:
            from sglang.srt.runtime_context import SpawnRanks, publish
            from sglang.srt.distributed.bootstrap import init_parallel_runtime
            publish(server_args, role='scheduler', ranks=SpawnRanks(world_rank=tp_rank, gpu_id=gpu_id))
            init_parallel_runtime(server_args=server_args, device=getattr(server_args, 'device', 'cuda') or 'cuda', dist_port=self.nccl_port)
        except Exception as _e:
            print(f'[audio8-tts] runtime_context setup warning: {_e}')
        super().__init__(**{k: v for k, v in _mr_kwargs.items() if k in _mr_params})
        if not hasattr(self, 'tp_group'):
            try:
                from sglang.srt.runtime_context import get_parallel
                self.tp_group = get_parallel().tp_group
            except Exception:
                pass'''
        txt = txt.replace(old_super, new_super, 1)
    if '# patched-by-audio8-tts' not in txt:
        marker = '    def _register_omni_model(self):'
        idx = txt.find(marker)
        if idx != -1:
            replacement = '''    def _register_omni_model(self):
        # patched-by-audio8-tts
        from sglang.srt.models.registry import ModelRegistry
        for _reg in (
            ('sglang_omni.models.fishaudio_s2_pro.sglang_model', 'S2ProSGLangTextModel'),
            ('sglang_omni.models.qwen3_omni.talker', 'Qwen3OmniTalker'),
            ('sglang_omni.models.ming_omni.thinker', 'BailingMoeV2ForCausalLM'),
        ):
            try:
                _mod = __import__(_reg[0], fromlist=[_reg[1]])
                ModelRegistry.models[_reg[1]] = getattr(_mod, _reg[1])
            except Exception:
                pass
        try:
            from sglang_omni.models.ming_omni.thinker import BailingMM2Config
            from transformers import AutoConfig
            AutoConfig.register(\"bailingmm_moe_v2_lite\", BailingMM2Config)
        except Exception:
            pass
'''
            txt = txt[:idx] + replacement
    mr.write_text(txt)
" || true

# Patch sglang_omni/engines/ar/sglang_backend/scheduler/cache.py when ChunkCache is removed in newer sglang
"${PYTHON_BIN}" -c "
import pathlib
p = pathlib.Path('/opt/sglang-omni/sglang_omni/engines/ar/sglang_backend/scheduler/cache.py')
if p.exists():
    txt = p.read_text()
    old = '        from sglang.srt.mem_cache.chunk_cache import ChunkCache'
    new = '''        try:
            from sglang.srt.mem_cache.chunk_cache import ChunkCache
        except ImportError:
            from sglang.srt.mem_cache.radix_cache import RadixCache as ChunkCache'''
    if old in txt and 'except ImportError:' not in txt:
        p.write_text(txt.replace(old, new, 1))
" || true

# Patch OmniEngine._run_loop so exceptions in schedule() are logged instead of silently killing the task
"${PYTHON_BIN}" -c "
import pathlib
p = pathlib.Path('/opt/sglang-omni/sglang_omni/engines/omni/engine.py')
if p.exists():
    txt = p.read_text()
    old = '''        while self._running:
            if self.enable_overlap:
                await self._step_overlap()
            else:
                await self._step_normal()
            await asyncio.sleep(0)  # Yield to other coroutines'''
    new = '''        while self._running:
            try:
                if self.enable_overlap:
                    await self._step_overlap()
                else:
                    await self._step_normal()
            except Exception:
                logger.exception(\"OmniEngine loop error\")
                await asyncio.sleep(0.05)
            await asyncio.sleep(0)  # Yield to other coroutines'''
    if old in txt and 'OmniEngine loop error' not in txt:
        p.write_text(txt.replace(old, new, 1))
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
