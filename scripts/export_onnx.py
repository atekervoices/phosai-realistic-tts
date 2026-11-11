#!/usr/bin/env python3
"""
Memory-isolated ONNX exporter and 94-speaker voice pre-encoder for `phosai/realistic-v1`
(Audio8-TTS-Realistic-Air 0.6B).

Runs heavy stages in isolated child subprocesses so the OS kernel reclaims 100% of RAM
between stages (preventing WSL2 / Docker build OOM kills):
  1. `--stage voices-and-codec`: Drops the 0.6B Transformer weights before loading the
     audio codec, pre-encodes all 94 speaker references (capped to 10s), and exports
     `codec_decoder_fp32.onnx` + `registration/codec_encoder.onnx`.
  2. `--stage fast-ar`: Drops the 28 Slow AR layers and exports `fast_ar_fp32.onnx`.
  3. `--stage slow-ar`: Drops the Fast AR layers and traces `slow_ar_fp32.onnx` using
     a lightweight dynamic KV-cache length (`dummy_cache_len=16`).
  4. `--stage quantize`: Runs in a clean non-PyTorch process to quantize `fast_ar` and
     `slow_ar` to INT8 and write `runtime_manifest.json`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Replace torch.inference_mode with torch.no_grad before importing dynamic HF modules
# so @torch.inference_mode() decorators in modeling_arktts*.py do not break TorchScript ONNX tracing.
torch.inference_mode = torch.no_grad  # type: ignore[assignment]


class _GlobalSafePolar:
    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a_f = a.float() if isinstance(a, torch.Tensor) else float(a)
        b_f = b.float() if isinstance(b, torch.Tensor) else float(b)
        self.abs = a_f
        self.angle = b_f
        self.real = a_f * torch.cos(b_f)
        self.imag = a_f * torch.sin(b_f)

    def to(self, *args: Any, **kwargs: Any) -> "_GlobalSafePolar":
        self.real = self.real.to(*args, **kwargs)
        self.imag = self.imag.to(*args, **kwargs)
        return self

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.real if idx == 0 else self.imag


_orig_torch_polar = getattr(torch, "polar", None)


def _monkey_torch_polar(a: torch.Tensor, b: torch.Tensor) -> _GlobalSafePolar:
    return _GlobalSafePolar(a, b)


torch.polar = _monkey_torch_polar  # type: ignore[assignment]


def register_custom_symbolics() -> None:
    from torch.onnx import register_custom_op_symbolic
    import torch.onnx.symbolic_opset9 as sym9

    def _symbolic_iand(g, *inputs, **kwargs):
        self_val = inputs[0] if len(inputs) > 0 else None
        other_val = inputs[1] if len(inputs) > 1 else None
        try:
            if hasattr(sym9, "bitwise_and"):
                return sym9.bitwise_and(g, self_val, other_val)
            if hasattr(sym9, "__and_"):
                return sym9.__and_(g, self_val, other_val)
        except Exception:
            pass
        try:
            if hasattr(self_val, "type") and self_val.type() is not None and self_val.type().scalarType() == "Bool":
                return g.op("And", self_val, other_val)
        except Exception:
            pass
        try:
            return g.op("BitwiseAnd", self_val, other_val)
        except Exception:
            return g.op("And", self_val, other_val)

    def _symbolic_ior(g, *inputs, **kwargs):
        self_val = inputs[0] if len(inputs) > 0 else None
        other_val = inputs[1] if len(inputs) > 1 else None
        try:
            if hasattr(sym9, "bitwise_or"):
                return sym9.bitwise_or(g, self_val, other_val)
            if hasattr(sym9, "__or_"):
                return sym9.__or_(g, self_val, other_val)
        except Exception:
            pass
        try:
            return g.op("BitwiseOr", self_val, other_val)
        except Exception:
            return g.op("Or", self_val, other_val)

    def _symbolic_ixor(g, *inputs, **kwargs):
        self_val = inputs[0] if len(inputs) > 0 else None
        other_val = inputs[1] if len(inputs) > 1 else None
        try:
            if hasattr(sym9, "bitwise_xor"):
                return sym9.bitwise_xor(g, self_val, other_val)
            if hasattr(sym9, "__xor_"):
                return sym9.__xor_(g, self_val, other_val)
        except Exception:
            pass
        try:
            return g.op("BitwiseXor", self_val, other_val)
        except Exception:
            return g.op("Xor", self_val, other_val)

    def _symbolic_polar(g, abs_val, angle_val):
        angle_float = g.op("Cast", angle_val, to_i=1)
        abs_float = g.op("Cast", abs_val, to_i=1)
        cos_val = g.op("Cos", angle_float)
        sin_val = g.op("Sin", angle_float)
        real_val = g.op("Mul", abs_float, cos_val)
        imag_val = g.op("Mul", abs_float, sin_val)
        dim_const = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        real_u = g.op("Unsqueeze", real_val, dim_const)
        imag_u = g.op("Unsqueeze", imag_val, dim_const)
        return g.op("Concat", real_u, imag_u, axis=-1)

    for opset in range(9, 21):
        for prefix in ["aten::", "::aten::"]:
            for op_name, fn in [
                ("__iand_", _symbolic_iand),
                ("__ior_", _symbolic_ior),
                ("__ixor_", _symbolic_ixor),
                ("polar", _symbolic_polar),
            ]:
                try:
                    register_custom_op_symbolic(f"{prefix}{op_name}", fn, opset_version=opset)
                except Exception:
                    pass

    for mod_name in [
        "symbolic_opset9",
        "symbolic_opset10",
        "symbolic_opset11",
        "symbolic_opset12",
        "symbolic_opset13",
        "symbolic_opset14",
        "symbolic_opset15",
        "symbolic_opset16",
        "symbolic_opset17",
        "symbolic_opset18",
        "symbolic_opset19",
        "symbolic_opset20",
    ]:
        try:
            mod = getattr(__import__("torch.onnx", fromlist=[mod_name]), mod_name)
            setattr(mod, "__iand_", _symbolic_iand)
            setattr(mod, "__ior_", _symbolic_ior)
            setattr(mod, "__ixor_", _symbolic_ixor)
            setattr(mod, "polar", _symbolic_polar)
        except Exception:
            pass


register_custom_symbolics()


def patch_source_files(src_path: Path) -> None:
    """
    Patch downloaded model python files to convert in-place bitwise operators (&=, |=, ^=)
    and complex torch.polar calls into clean real arithmetic, preventing TorchScript from
    generating unsupported JIT nodes like aten::__iand_ and aten::polar.
    """
    import re

    if not src_path.is_dir():
        return
    for py_file in src_path.glob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            orig = content
            if "&=" in content:
                content = re.sub(
                    r"^([ \t]*)([a-zA-Z0-9_\.\[\]]+)\s*&=\s*(.+)$",
                    r"\1\2 = \2 & (\3)",
                    content,
                    flags=re.MULTILINE,
                )
            if "|=" in content:
                content = re.sub(
                    r"^([ \t]*)([a-zA-Z0-9_\.\[\]]+)\s*\|=\s*(.+)$",
                    r"\1\2 = \2 | (\3)",
                    content,
                    flags=re.MULTILINE,
                )
            if "^=" in content:
                content = re.sub(
                    r"^([ \t]*)([a-zA-Z0-9_\.\[\]]+)\s*\^=\s*(.+)$",
                    r"\1\2 = \2 ^ (\3)",
                    content,
                    flags=re.MULTILINE,
                )
            if "def _extra_padding" in content:
                print(f"Patching _extra_padding in {py_file.name} for dynamic ONNX tracing...")
                content = re.sub(
                    r"def _extra_padding\(x, kernel_size: int, stride: int, padding_total: int = 0\) -> int:[\s\S]*?return ideal - length",
                    (
                        "def _extra_padding(x, kernel_size: int, stride: int, padding_total: int = 0):\n"
                        "    if stride == 1:\n"
                        "        return 0\n"
                        "    length = x.shape[-1]\n"
                        "    rem = (length - kernel_size + padding_total) % stride\n"
                        "    return (stride - rem) % stride"
                    ),
                    content,
                )
            if "class ArkttsConvNeXtBlock" in content and "min_len" not in content:
                content = content.replace(
                    "return residual + x",
                    "min_len = min(residual.shape[-1], x.shape[-1])\n        return residual[..., :min_len] + x[..., :min_len]",
                )
            if "class ArkttsResidualUnit" in content and "min_len" not in content:
                content = content.replace(
                    "return x + output",
                    "min_len = min(x.shape[-1], output.shape[-1])\n        return x[..., :min_len] + output[..., :min_len]",
                )
            if "def _rope(" in content:
                print(f"Patching _rope in {py_file.name} for ONNX export compatibility...")
                content = re.sub(
                    r"def _rope\(length: int, head_dim: int, base: float, device=None\) -> Tensor:[\s\S]*?return torch\.stack\(\(values\.real, values\.imag\), dim=-1\)\.to\(torch\.bfloat16\)",
                    (
                        "def _rope(length: int, head_dim: int, base: float, device=None) -> Tensor:\n"
                        "    frequencies = 1.0 / (\n"
                        "        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)\n"
                        "    )\n"
                        "    t = torch.arange(length, device=device, dtype=torch.float32)\n"
                        "    phases = torch.outer(t, frequencies)\n"
                        "    cos = torch.cos(phases).float()\n"
                        "    sin = torch.sin(phases).float()\n"
                        "    return torch.stack((cos, sin), dim=-1).float()"
                    ),
                    content,
                )
            if "torch.polar" in content or "_safe_polar" in content:
                print(f"Patching torch.polar in {py_file.name} for ONNX export compatibility...")
                safe_polar_code = (
                    "\nclass _SafePolar:\n"
                    "    def __init__(self, a, b):\n"
                    "        a_f = a.float() if hasattr(a, 'float') else a\n"
                    "        b_f = b.float() if hasattr(b, 'float') else b\n"
                    "        self.real = a_f * torch.cos(b_f)\n"
                    "        self.imag = a_f * torch.sin(b_f)\n"
                    "    def to(self, *args, **kwargs):\n"
                    "        self.real = self.real.to(*args, **kwargs)\n"
                    "        self.imag = self.imag.to(*args, **kwargs)\n"
                    "        return self\n"
                    "    def __getitem__(self, idx):\n"
                    "        return self.real if idx == 0 else self.imag\n"
                    "def _safe_polar(a, b):\n"
                    "    return _SafePolar(a, b)\n"
                )
                if "_safe_polar" not in content:
                    content = content + "\n" + safe_polar_code
                content = content.replace("torch.polar(", "_safe_polar(")

            if "from __future__" in content:
                # Ensure all `from __future__ import ...` statements are strictly at the beginning of the file
                future_lines = [l for l in content.splitlines() if l.startswith("from __future__")]
                other_lines = [l for l in content.splitlines() if not l.startswith("from __future__")]
                content = "\n".join(future_lines + [""] + other_lines)
            if content != orig:
                print(f"Patched compatibility fixes in {py_file.name} for ONNX export.")
                py_file.write_text(content, encoding="utf-8")
        except Exception as exc:
            print(f"[Warn] Could not patch {py_file}: {exc}")


from huggingface_hub import snapshot_download
from scipy.signal import resample_poly
from transformers import AutoConfig, AutoModel

DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "phosai/realistic-v1")
MAX_REF_AUDIO_SECONDS = 10.0


def prepare_module_for_onnx(module: nn.Module) -> nn.Module:
    """
    Fuse any weight_norm parametrizations and detach/freeze all parameters, buffers,
    and plain tensor attributes so TorchScript tracing never encounters requires_grad=True.
    """
    for sub in list(module.modules()):
        try:
            torch.nn.utils.remove_weight_norm(sub)
        except Exception:
            pass
        try:
            if hasattr(torch.nn.utils, "parametrize") and torch.nn.utils.parametrize.is_parametrized(sub):
                for param_name in list(sub.parametrizations.keys()):
                    torch.nn.utils.parametrize.remove_parametrizations(
                        sub, param_name, leave_parametrized=True
                    )
        except Exception:
            pass

    module.eval().float().requires_grad_(False)
    for p in module.parameters():
        p.data = p.detach().float().clone()
        p.requires_grad_(False)
    for b in module.buffers():
        b.data = b.detach().float().clone() if b.is_floating_point() else b.detach().clone()
        b.requires_grad_(False)
    for sub in module.modules():
        for k, v in list(sub.__dict__.items()):
            if isinstance(v, torch.Tensor):
                detached = v.detach().float().clone() if v.is_floating_point() else v.detach().clone()
                detached.requires_grad_(False)
                setattr(sub, k, detached)
    return module


def load_arktts_codec(src_path: Path, device: str = "cpu") -> nn.Module:
    """
    Load ArkttsCodec directly from `modeling_arktts_codec.py` and `codec.pth`
    without instantiating the 0.6B Slow/Fast Transformer.
    """
    patch_source_files(src_path)
    cfg_dict = json.loads((src_path / "config.json").read_text(encoding="utf-8"))
    cfg = SimpleNamespace(**cfg_dict)
    module_path = src_path / "modeling_arktts_codec.py"
    spec = importlib.util.spec_from_file_location("_audio8_arktts_codec", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Audio8 codec from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    codec = mod.ArkttsCodec(cfg)
    state = torch.load(
        src_path / "codec.pth",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {
            key.replace("generator.", ""): value
            for key, value in state.items()
            if "generator." in key
        }
    state = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
        if not key.endswith(("freqs_cis", "causal_mask"))
    }
    codec.load_state_dict(state, strict=True, assign=True)
    codec = codec.eval().to(device=device, dtype=torch.float32)
    return prepare_module_for_onnx(codec)


def onnx_export(model: nn.Module, args: Tuple[Any, ...], path: str, **kwargs: Any) -> None:
    prepare_module_for_onnx(model)
    register_custom_symbolics()
    kwargs.setdefault("opset_version", 18)
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        kwargs.setdefault("dynamo", False)
    with torch.no_grad():
        torch.onnx.export(model, args, path, **kwargs)


def sanitize_dirname(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings (`freqs_cis` of shape `[L, head_dim // 2, 2]`)."""
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope = freqs_cis.float()
    if rope.ndim == 2:
        rope = torch.stack([torch.cos(rope), torch.sin(rope)], dim=-1)
    rope = rope.unsqueeze(0).unsqueeze(2)  # [1, L, 1, head_dim // 2, 2]
    out_0 = shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1]
    out_1 = shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1]
    return torch.stack((out_0, out_1), dim=-1).flatten(3).to(x.dtype)


class SlowAROnnxWrapper(nn.Module):
    """
    ONNX-traceable wrapper for Audio8 0.6B Slow AR transformer with dynamic KV-cache length.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval().float()
        cfg = model.config
        self.num_layers = int(cfg.n_layer)
        self.num_codebooks = int(cfg.num_codebooks)
        self.codebook_size = int(cfg.codebook_size)
        self.semantic_begin_id = int(cfg.semantic_begin_id)
        self.semantic_end_id = int(cfg.semantic_end_id)
        self.im_end_id = int(getattr(cfg, "eos_token_id", 151645))
        self.n_head = int(cfg.n_head)
        self.n_local_heads = int(cfg.n_local_heads)
        self.head_dim = int(cfg.head_dim)
        self.norm_fastlayer_input = bool(getattr(cfg, "norm_fastlayer_input", True))

        offsets = torch.arange(self.num_codebooks, dtype=torch.long) * self.codebook_size
        self.register_buffer("codebook_offsets", offsets.view(1, self.num_codebooks, 1))

    def forward(
        self,
        codes: torch.Tensor,
        input_pos: torch.Tensor,
        *kv_caches: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        row0 = codes[:, 0, :]
        vq_rows = codes[:, 1:, :]

        emb_layer = getattr(self.model, "embeddings", None) or getattr(
            self.model, "embed_tokens"
        )
        text_emb = emb_layer(row0)

        offset_vq = vq_rows + self.codebook_offsets
        cb_emb = self.model.codebook_embeddings(offset_vq).sum(dim=1)
        if getattr(self.model.config, "scale_codebook_embeddings", False):
            cb_emb = cb_emb / math.sqrt(self.num_codebooks + 1)
            text_emb_scaled = text_emb / math.sqrt(self.num_codebooks + 1)
        else:
            text_emb_scaled = text_emb

        is_semantic = (row0 >= self.semantic_begin_id) & (row0 <= self.semantic_end_id)
        hidden = torch.where(is_semantic.unsqueeze(-1), text_emb_scaled + cb_emb, text_emb)

        rope_cis = self.model.freqs_cis[input_pos]
        seq_len = codes.shape[2]
        cache_len = kv_caches[0].shape[2]
        cache_indices = torch.arange(cache_len, device=codes.device).unsqueeze(0)
        valid_mask = cache_indices <= input_pos.unsqueeze(1)
        attn_bias = torch.where(valid_mask, 0.0, -1e9).view(1, 1, seq_len, cache_len)

        next_keys: List[torch.Tensor] = []
        next_values: List[torch.Tensor] = []

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        repeat_factor = self.n_head // self.n_local_heads

        for idx, layer in enumerate(self.model.layers):
            k_cache = kv_caches[idx]
            v_cache = kv_caches[self.num_layers + idx]

            normed = layer.attention_norm(hidden)
            attn = layer.attention
            q, k, v = attn.wqkv(normed).split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(1, seq_len, self.n_head, self.head_dim)
            k = k.view(1, seq_len, self.n_local_heads, self.head_dim)
            v = v.view(1, seq_len, self.n_local_heads, self.head_dim)

            if getattr(attn, "qk_norm", False) and hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            q = apply_rope(q, rope_cis).transpose(1, 2)
            k = apply_rope(k, rope_cis).transpose(1, 2)
            v = v.transpose(1, 2)

            next_keys.append(k)
            next_values.append(v)

            pos_idx = input_pos.view(1, 1, seq_len, 1).expand_as(k)
            updated_k = k_cache.scatter(2, pos_idx, k)
            updated_v = v_cache.scatter(2, pos_idx, v)

            k_rep = updated_k.repeat_interleave(repeat_factor, dim=1)
            v_rep = updated_v.repeat_interleave(repeat_factor, dim=1)

            scores = torch.matmul(q, k_rep.transpose(-2, -1)) * (self.head_dim**-0.5)
            scores = scores + attn_bias
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v_rep)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, q_size)

            hidden = hidden + attn.wo(attn_out)
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

        normed_hidden = self.model.norm(hidden)
        fast_hidden = normed_hidden if self.norm_fastlayer_input else hidden

        head_layer = (
            getattr(self.model, "output", None)
            or getattr(self.model, "semantic_output", None)
            or getattr(self.model, "lm_head", None)
            or emb_layer
        )
        head_weight = head_layer.weight
        if head_weight.shape[0] == self.codebook_size + 1:
            logits = F.linear(normed_hidden, head_weight, getattr(head_layer, "bias", None))
        else:
            sem_weight = head_weight[self.semantic_begin_id : self.semantic_end_id + 1]
            eos_weight = head_weight[self.im_end_id : self.im_end_id + 1]
            proj_weight = torch.cat([sem_weight, eos_weight], dim=0)
            logits = F.linear(normed_hidden, proj_weight)

        return (logits, fast_hidden, *next_keys, *next_values)


class FastAROnnxWrapper(nn.Module):
    """
    ONNX-traceable wrapper for Audio8 0.6B Fast AR 4-layer codebook head.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval().float()
        cfg = model.config
        self.num_fast_layers = int(cfg.n_fast_layer)
        self.num_codebooks = int(cfg.num_codebooks)
        self.n_head = int(cfg.fast_n_head)
        self.n_local_heads = int(cfg.fast_n_local_heads)
        self.head_dim = int(cfg.fast_head_dim)

    def forward(
        self,
        slow_hidden: torch.Tensor,
        token_id: torch.Tensor,
        use_slow_hidden: torch.Tensor,
        input_pos: torch.Tensor,
        *fast_caches: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        proj_slow = self.model.fast_project_in(slow_hidden)
        tok_emb = self.model.fast_embeddings(token_id)
        hidden = torch.where(use_slow_hidden.view(1, 1, 1), proj_slow, tok_emb)

        rope_cis = self.model.fast_freqs_cis[input_pos]
        cache_indices = torch.arange(self.num_codebooks, device=token_id.device).unsqueeze(0)
        valid_mask = cache_indices <= input_pos.unsqueeze(1)
        attn_bias = torch.where(valid_mask, 0.0, -1e9).view(1, 1, 1, self.num_codebooks)

        next_keys: List[torch.Tensor] = []
        next_values: List[torch.Tensor] = []

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        repeat_factor = self.n_head // self.n_local_heads

        for idx, layer in enumerate(self.model.fast_layers):
            k_cache = fast_caches[idx]
            v_cache = fast_caches[self.num_fast_layers + idx]

            normed = layer.attention_norm(hidden)
            attn = layer.attention
            q, k, v = attn.wqkv(normed).split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(1, 1, self.n_head, self.head_dim)
            k = k.view(1, 1, self.n_local_heads, self.head_dim)
            v = v.view(1, 1, self.n_local_heads, self.head_dim)

            if getattr(attn, "qk_norm", False) and hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            q = apply_rope(q, rope_cis).transpose(1, 2)
            k = apply_rope(k, rope_cis).transpose(1, 2)
            v = v.transpose(1, 2)

            next_keys.append(k)
            next_values.append(v)

            pos_idx = input_pos.view(1, 1, 1, 1).expand_as(k)
            updated_k = k_cache.scatter(2, pos_idx, k)
            updated_v = v_cache.scatter(2, pos_idx, v)

            k_rep = updated_k.repeat_interleave(repeat_factor, dim=1)
            v_rep = updated_v.repeat_interleave(repeat_factor, dim=1)

            scores = torch.matmul(q, k_rep.transpose(-2, -1)) * (self.head_dim**-0.5)
            scores = scores + attn_bias
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v_rep)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, q_size)

            hidden = hidden + attn.wo(attn_out)
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

        logits = self.model.fast_output(self.model.fast_norm(hidden))
        return (logits, *next_keys, *next_values)


class CodecDecoderOnnxWrapper(nn.Module):
    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        decode_method = getattr(self.codec, "decode", None)
        if hasattr(decode_method, "__wrapped__"):
            out = decode_method.__wrapped__(self.codec, codes)
        else:
            out = self.codec.decode(codes)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim == 3:
            out = out.squeeze(1)
        elif out.ndim == 1:
            out = out.unsqueeze(0)
        return out


class CodecEncoderOnnxWrapper(nn.Module):
    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        wav_2d = audio.squeeze(1) if audio.ndim == 3 else audio
        lengths = torch.tensor([wav_2d.shape[-1]], dtype=torch.long, device=audio.device)
        encode_method = getattr(self.codec, "encode", None)
        if hasattr(encode_method, "__wrapped__"):
            res = encode_method.__wrapped__(self.codec, wav_2d, lengths)
        else:
            res = self.codec.encode(wav_2d, lengths)
        indices = res[0] if isinstance(res, tuple) else res
        return indices.to(torch.int64)


def get_persistent_cache_dir(fingerprint: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / f"audio8_build_cache_{fingerprint}_v5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def pre_encode_speaker_references(
    codec: nn.Module,
    registry_path: Path,
    voices_dir: Path,
    sample_rate: int,
    hop_length: int,
    model_fingerprint: str,
    device: str,
) -> None:
    if not registry_path.is_file():
        print(f"No speaker_registry.json found at {registry_path}, skipping pre-encoding.")
        return

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    voices_dir.mkdir(parents=True, exist_ok=True)
    voices_cache_dir = get_persistent_cache_dir(model_fingerprint) / "voices"
    voices_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    v1_voices_dir = hf_home / f"audio8_build_cache_{model_fingerprint}" / "voices"

    base_dir = registry_path.parent
    total = len(registry)
    print(f"Pre-encoding {total} speaker references into {voices_dir} (max {MAX_REF_AUDIO_SECONDS}s per speaker)...")

    max_samples = int(MAX_REF_AUDIO_SECONDS * sample_rate)
    encoded_ok = 0
    restored_from_cache = 0

    for idx, (full_key, entry) in enumerate(registry.items(), start=1):
        safe_stem = full_key.replace(":", "__")
        rust_json_path = voices_dir / f"{safe_stem}.json"
        safe_name = sanitize_dirname(full_key)
        target_dir = voices_dir / safe_name
        codes_file = target_dir / "codes.npy"
        meta_file = target_dir / "meta.json"

        cached_json = voices_cache_dir / f"{safe_stem}.json"
        cached_codes = voices_cache_dir / safe_name / "codes.npy"
        cached_meta = voices_cache_dir / safe_name / "meta.json"

        # Check v2 or v1 cache if not yet in v3 cache
        if not cached_json.is_file():
            for v_dir in [
                hf_home / f"audio8_build_cache_{model_fingerprint}_v2" / "voices",
                hf_home / f"audio8_build_cache_{model_fingerprint}" / "voices",
            ]:
                if v_dir.is_dir():
                    v_json = v_dir / f"{safe_stem}.json"
                    v_c = v_dir / safe_name / "codes.npy"
                    v_m = v_dir / safe_name / "meta.json"
                    if v_json.is_file() and v_c.is_file() and v_m.is_file():
                        cached_codes.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(v_json, cached_json)
                        shutil.copy2(v_c, cached_codes)
                        shutil.copy2(v_m, cached_meta)
                        break

        if rust_json_path.is_file() and codes_file.is_file():
            encoded_ok += 1
            continue

        if cached_json.is_file() and cached_codes.is_file() and cached_meta.is_file():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_json, rust_json_path)
            shutil.copy2(cached_codes, codes_file)
            shutil.copy2(cached_meta, meta_file)
            encoded_ok += 1
            restored_from_cache += 1
            continue

        wav_rel = Path(entry.get("reference_audio") or entry.get("audio_path", ""))
        wav_path = wav_rel if wav_rel.is_file() else (base_dir / wav_rel)
        if not wav_path.is_file():
            print(f"  [Skip] Missing reference WAV: {wav_path}")
            continue

        audio_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        mono = audio_np.mean(axis=1)
        if int(sr) != sample_rate:
            g = math.gcd(int(sr), sample_rate)
            mono = resample_poly(mono, sample_rate // g, int(sr) // g).astype(np.float32)

        if mono.size > max_samples:
            mono = mono[:max_samples]

        remainder = mono.size % hop_length
        if remainder:
            mono = np.pad(mono, (0, hop_length - remainder))

        with torch.no_grad():
            wav_2d = torch.from_numpy(mono).unsqueeze(0).to(device=device, dtype=torch.float32)
            lengths = torch.tensor([mono.size], device=device, dtype=torch.long)
            indices, feature_lens = codec.encode(wav_2d, lengths)

            n_frames = int(feature_lens[0].item())
            codes_np = np.ascontiguousarray(
                indices[0, :, :n_frames].detach().cpu().numpy().astype(np.int64)
            )
            del indices, feature_lens

        target_dir.mkdir(parents=True, exist_ok=True)
        np.save(codes_file, codes_np, allow_pickle=False)
        ref_txt = entry.get("reference_text", "").strip()
        meta = {
            "name": full_key,
            "reference_text": ref_txt,
            "source_audio": str(wav_rel),
            "sample_rate": sample_rate,
            "shape": list(codes_np.shape),
            "model_fingerprint": model_fingerprint,
        }
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        short_id = full_key.split(":", 1)[1] if ":" in full_key else full_key
        lang_code = short_id.split("_", 1)[0] if "_" in short_id else "eng"
        rust_profile = {
            "name": full_key,
            "language": lang_code,
            "reference_text": ref_txt,
            "audio_path": str(wav_rel),
            "vq_codes": codes_np.tolist(),
        }
        rust_json_path.write_text(
            json.dumps(rust_profile, ensure_ascii=False), encoding="utf-8"
        )

        # Persist in BuildKit cache volume so rebuilds restore in <0.1s
        cached_codes.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rust_json_path, cached_json)
        shutil.copy2(codes_file, cached_codes)
        shutil.copy2(meta_file, cached_meta)

        encoded_ok += 1
        if idx % 10 == 0 or idx == total:
            gc.collect()
            print(f"  Encoded [{idx}/{total}] speaker voices...")

    print(
        f"Successfully prepared {encoded_ok}/{total} speaker voices in {voices_dir} "
        f"({restored_from_cache} restored from cache)!"
    )


def run_stage_voices_and_codec(src_path: Path, out_dir: Path, voices_dir: Path, registry_path: Path, model_id: str) -> None:
    print("=== [Stage 1/4] Exporting Codec ONNX & pre-encoding 94 voices (Direct ArkttsCodec, no LLM loaded) ===")
    cfg = json.loads((src_path / "config.json").read_text(encoding="utf-8"))
    num_cb = int(cfg["num_codebooks"])
    sample_rate = int(cfg.get("codec_sample_rate", cfg.get("sample_rate", 44100)))
    hop_length = int(cfg.get("codec_hop_length", 2048))
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]:
        if (src_path / tok_file).is_file():
            shutil.copy2(src_path / tok_file, tok_dir / tok_file)
            shutil.copy2(src_path / tok_file, out_dir / tok_file)

    codec = load_arktts_codec(src_path, device="cpu")

    model_cache_dir = get_persistent_cache_dir(fingerprint) / "model"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_dec = model_cache_dir / "codec_decoder_fp32.onnx"
    cached_enc = model_cache_dir / "codec_encoder.onnx"

    dec_onnx_path = out_dir / "codec_decoder_fp32.onnx"
    if cached_dec.is_file():
        print(f"Restoring {dec_onnx_path.name} from BuildKit persistent cache...")
        shutil.copy2(cached_dec, dec_onnx_path)
    else:
        dec_wrapper = CodecDecoderOnnxWrapper(codec)
        dummy_vq = torch.zeros((1, num_cb, 8), dtype=torch.long)
        print(f"Exporting Codec Decoder to {dec_onnx_path}...")
        onnx_export(
            dec_wrapper,
            (dummy_vq,),
            str(dec_onnx_path),
            input_names=["codes"],
            output_names=["audio"],
            dynamic_axes={"codes": {2: "frames"}, "audio": {1: "samples"}},
            opset_version=18,
            do_constant_folding=True,
        )
        print(f"Validating {dec_onnx_path} across dynamic lengths with ONNX Runtime...")
        import onnxruntime as ort
        _test_sess = ort.InferenceSession(str(dec_onnx_path), providers=["CPUExecutionProvider"])
        for test_len in [4, 8, 11, 23, 50]:
            _dummy = np.zeros((1, num_cb, test_len), dtype=np.int64)
            _res = _test_sess.run(None, {"codes": _dummy})
            print(f"  Dynamic length {test_len} decoded successfully -> audio shape {_res[0].shape}")
        del _test_sess
        shutil.copy2(dec_onnx_path, cached_dec)

    reg_dir = out_dir / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)
    enc_onnx_path = reg_dir / "codec_encoder.onnx"
    if cached_enc.is_file():
        print(f"Restoring {enc_onnx_path.name} from BuildKit persistent cache...")
        shutil.copy2(cached_enc, enc_onnx_path)
    else:
        enc_wrapper = CodecEncoderOnnxWrapper(codec)
        dummy_wav = torch.zeros((1, 1, hop_length * 8), dtype=torch.float32)
        print(f"Exporting Codec Encoder to {enc_onnx_path}...")
        try:
            onnx_export(
                enc_wrapper,
                (dummy_wav,),
                str(enc_onnx_path),
                input_names=["audio"],
                output_names=["codes"],
                dynamic_axes={"audio": {2: "samples"}, "codes": {2: "frames"}},
                opset_version=18,
                do_constant_folding=True,
            )
            shutil.copy2(enc_onnx_path, cached_enc)
        except Exception as exc:
            print(f"[Warn] Optional codec_encoder.onnx export skipped ({exc}); pre-encoded voices do not require it.")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str != "cpu":
        codec = codec.to(device=device_str, dtype=torch.float32)

    pre_encode_speaker_references(
        codec=codec,
        registry_path=registry_path,
        voices_dir=voices_dir,
        sample_rate=sample_rate,
        hop_length=hop_length,
        model_fingerprint=fingerprint,
        device=device_str,
    )


def run_stage_fast_ar(src_path: Path, out_dir: Path, model_id: str = DEFAULT_MODEL_ID) -> None:
    print("=== [Stage 2/4] Exporting Fast AR ONNX (Slow AR & Codec unloaded) ===")
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]
    model_cache_dir = get_persistent_cache_dir(fingerprint) / "model"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_fast = model_cache_dir / "fast_ar_fp32.onnx"
    fast_onnx_path = out_dir / "fast_ar_fp32.onnx"

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if not cached_fast.is_file():
        for prev_cache in [
            hf_home / f"audio8_build_cache_{fingerprint}_v2" / "model" / "fast_ar_fp32.onnx",
            hf_home / f"audio8_build_cache_{fingerprint}" / "model" / "fast_ar_fp32.onnx",
        ]:
            if prev_cache.is_file():
                shutil.copy2(prev_cache, cached_fast)
                break

    if cached_fast.is_file():
        print(f"Restoring {fast_onnx_path.name} from BuildKit persistent cache...")
        shutil.copy2(cached_fast, fast_onnx_path)
        return

    model = AutoModel.from_pretrained(
        str(src_path),
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    cfg = model.config

    # Free the 28 Slow AR layers immediately to save ~2.2 GB RAM
    for attr in ["layers", "embeddings", "embed_tokens", "codebook_embeddings", "output", "semantic_output", "lm_head", "norm"]:
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()

    fast_wrapper = FastAROnnxWrapper(model)
    num_fast = int(cfg.n_fast_layer)
    fast_heads = int(cfg.fast_n_local_heads)
    fast_head_dim = int(cfg.fast_head_dim)
    num_cb = int(cfg.num_codebooks)
    dim = int(cfg.dim)

    dummy_hidden = torch.zeros((1, 1, dim), dtype=torch.float32)
    dummy_tok = torch.zeros((1, 1), dtype=torch.long)
    dummy_use_slow = torch.tensor([True], dtype=torch.bool)
    dummy_fast_pos = torch.tensor([0], dtype=torch.long)
    dummy_fast_k = [
        torch.zeros((1, fast_heads, num_cb, fast_head_dim), dtype=torch.float32)
        for _ in range(num_fast)
    ]
    dummy_fast_v = [
        torch.zeros((1, fast_heads, num_cb, fast_head_dim), dtype=torch.float32)
        for _ in range(num_fast)
    ]

    fast_input_names = (
        ["slow_hidden", "token_id", "use_slow_hidden", "input_pos"]
        + [f"cache_key_{i}" for i in range(num_fast)]
        + [f"cache_value_{i}" for i in range(num_fast)]
    )
    fast_output_names = (
        ["logits"]
        + [f"next_cache_key_{i}" for i in range(num_fast)]
        + [f"next_cache_value_{i}" for i in range(num_fast)]
    )

    print(f"Exporting Fast AR to {fast_onnx_path}...")
    onnx_export(
        fast_wrapper,
        (dummy_hidden, dummy_tok, dummy_use_slow, dummy_fast_pos, *dummy_fast_k, *dummy_fast_v),
        str(fast_onnx_path),
        input_names=fast_input_names,
        output_names=fast_output_names,
        opset_version=18,
        do_constant_folding=True,
    )
    shutil.copy2(fast_onnx_path, cached_fast)


def run_stage_slow_ar(src_path: Path, out_dir: Path, model_id: str = DEFAULT_MODEL_ID) -> None:
    print("=== [Stage 3/4] Exporting Slow AR ONNX (Fast AR & Codec unloaded) ===")
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]
    model_cache_dir = get_persistent_cache_dir(fingerprint) / "model"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_slow = model_cache_dir / "slow_ar_fp32.onnx"
    slow_onnx_path = out_dir / "slow_ar_fp32.onnx"

    def copy_model_artifacts(src_d: Path, dst_d: Path, prefix: str) -> None:
        dst_d.mkdir(parents=True, exist_ok=True)
        for p in src_d.iterdir():
            if p.is_file() and (p.name.startswith(prefix) or p.name.endswith(".weight") or p.name.endswith(".data")):
                shutil.copy2(p, dst_d / p.name)

    def is_onnx_loadable(onnx_file: Path) -> bool:
        if not onnx_file.is_file():
            return False
        try:
            import onnxruntime as ort
            _sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
            return True
        except Exception as exc:
            print(f"[Warn] ONNX session verification failed for {onnx_file.name}: {exc}")
            return False

    if cached_slow.is_file() and is_onnx_loadable(cached_slow):
        print(f"Restoring {slow_onnx_path.name} & weights from BuildKit persistent cache...")
        copy_model_artifacts(model_cache_dir, out_dir, "slow_ar_fp32")
        if is_onnx_loadable(slow_onnx_path):
            return
        print("[Warn] Restored slow AR ONNX failed validation in target dir; re-exporting...")

    model = AutoModel.from_pretrained(
        str(src_path),
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    cfg = model.config

    # Free Fast AR layers before tracing Slow AR
    for attr in ["fast_layers", "fast_embeddings", "fast_output", "fast_project_in", "fast_norm"]:
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()

    slow_wrapper = SlowAROnnxWrapper(model)
    num_layers = int(cfg.n_layer)
    n_local_heads = int(cfg.n_local_heads)
    head_dim = int(cfg.head_dim)
    num_cb = int(cfg.num_codebooks)

    dummy_cache_len = 16
    dummy_codes = torch.zeros((1, num_cb + 1, 4), dtype=torch.long)
    dummy_pos = torch.arange(4, dtype=torch.long)
    dummy_slow_k = [
        torch.zeros((1, n_local_heads, dummy_cache_len, head_dim), dtype=torch.float32)
        for _ in range(num_layers)
    ]
    dummy_slow_v = [
        torch.zeros((1, n_local_heads, dummy_cache_len, head_dim), dtype=torch.float32)
        for _ in range(num_layers)
    ]

    slow_input_names = (
        ["codes", "input_pos"]
        + [f"cache_key_{i}" for i in range(num_layers)]
        + [f"cache_value_{i}" for i in range(num_layers)]
    )
    slow_output_names = (
        ["logits", "hidden"]
        + [f"next_cache_key_{i}" for i in range(num_layers)]
        + [f"next_cache_value_{i}" for i in range(num_layers)]
    )
    slow_dynamic_axes = {
        "codes": {2: "seq_len"},
        "input_pos": {0: "seq_len"},
        "logits": {1: "seq_len"},
        "hidden": {1: "seq_len"},
    }
    for i in range(num_layers):
        slow_dynamic_axes[f"cache_key_{i}"] = {2: "cache_len"}
        slow_dynamic_axes[f"cache_value_{i}"] = {2: "cache_len"}
        slow_dynamic_axes[f"next_cache_key_{i}"] = {2: "seq_len"}
        slow_dynamic_axes[f"next_cache_value_{i}"] = {2: "seq_len"}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting Slow AR to {slow_onnx_path}...")
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(out_dir))
        onnx_export(
            slow_wrapper,
            (dummy_codes, dummy_pos, *dummy_slow_k, *dummy_slow_v),
            "slow_ar_fp32.onnx",
            input_names=slow_input_names,
            output_names=slow_output_names,
            dynamic_axes=slow_dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
        )
        copy_model_artifacts(out_dir, model_cache_dir, "slow_ar_fp32")
    finally:
        os.chdir(prev_cwd)

    if not is_onnx_loadable(slow_onnx_path):
        raise RuntimeError(f"Exported Slow AR model at {slow_onnx_path} failed ONNX Runtime verification.")


def run_stage_quantize(src_path: Path, out_dir: Path, model_id: str, precision: str) -> None:
    print(f"=== [Stage 4/4] Finalizing ONNX models ({precision}) & writing runtime_manifest.json ===")
    cfg = AutoConfig.from_pretrained(str(src_path), trust_remote_code=True)
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]

    slow_models = {
        "fp32": "slow_ar_fp32.onnx",
        "int8": "slow_ar_fp32.onnx",
    }
    fast_models = {
        "fp32": "fast_ar_fp32.onnx",
    }

    fast_fp32 = out_dir / "fast_ar_fp32.onnx"
    fast_int8 = out_dir / "fast_ar_int8.onnx"
    if fast_fp32.is_file():
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            print("Quantizing Fast AR to INT8...")
            quantize_dynamic(str(fast_fp32), str(fast_int8), weight_type=QuantType.QInt8)
            fast_models["int8"] = "fast_ar_int8.onnx"
            gc.collect()
        except Exception as exc:
            print(f"[Warn] Fast AR INT8 quantization skipped ({exc}); using FP32.")
            fast_models["int8"] = "fast_ar_fp32.onnx"
    else:
        fast_models["int8"] = "fast_ar_fp32.onnx"

    # Create convenience symlinks/copies so any loader looking for unversioned names finds them
    for src_name, dst_names in [
        ("slow_ar_fp32.onnx", ["slow_ar.onnx", "slow_ar_int8.onnx"]),
        ("fast_ar_fp32.onnx", ["fast_ar.onnx"]),
        ("codec_decoder_fp32.onnx", ["codec_decoder.onnx"]),
    ]:
        src = out_dir / src_name
        if src.is_file():
            for dst_name in dst_names:
                dst = out_dir / dst_name
                try:
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    dst.symlink_to(src.name)
                except Exception:
                    try:
                        import shutil
                        if not dst.exists():
                            shutil.copyfile(src, dst)
                    except Exception:
                        pass

    print("Slow AR (0.6B) kept in FP32 for maximum numerical accuracy and zero-OOM Docker builds.")

    default_prec = "fp32"
    available_precisions = ["fp32", "int8"]

    manifest = {
        "model_id": model_id,
        "model_fingerprint": fingerprint,
        "default_precision": default_prec,
        "available_precisions": available_precisions,
        "default_codec_precision": "fp32",
        "available_codec_precisions": ["fp32"],
        "slow_decode_models": slow_models,
        "fast_models": fast_models,
        "codec_models": {"fp32": "codec_decoder_fp32.onnx"},
        "codec_encoder_model": "registration/codec_encoder.onnx",
        "sample_rate": int(getattr(cfg, "codec_sample_rate", getattr(cfg, "sample_rate", 44100))),
        "num_codebooks": int(cfg.num_codebooks),
        "codebook_size": int(cfg.codebook_size),
        "codec_hop_length": int(getattr(cfg, "codec_hop_length", 2048)),
        "stream_context_frames": 128,
        "stream_guard_frames": 1,
        "num_layers": int(cfg.n_layer),
        "num_fast_layers": int(cfg.n_fast_layer),
        "n_local_heads": int(cfg.n_local_heads),
        "head_dim": int(cfg.head_dim),
        "fast_head_dim": int(cfg.fast_head_dim),
        "dim": int(cfg.dim),
        "max_seq_len": min(int(cfg.max_seq_len), 1024),
        "semantic_begin_id": int(cfg.semantic_begin_id),
        "semantic_end_id": int(cfg.semantic_end_id),
        "im_end_id": int(getattr(cfg, "eos_token_id", 151645)),
        "slow_logits_layout": "relative_semantic_then_eos",
    }
    manifest_path = out_dir / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export phosai/realistic-v1 to ONNX Runtime")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face repo ID or local path")
    parser.add_argument("--hf-token", default=DEFAULT_HF_TOKEN, help="Hugging Face access token")
    parser.add_argument("--output-dir", default="model", help="Target directory for ONNX models")
    parser.add_argument("--voices-dir", default="voices", help="Target directory for pre-encoded voices")
    parser.add_argument("--registry", default="speaker_registry.json", help="Path to speaker_registry.json")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "int8"], help="ONNX export precision")
    parser.add_argument("--codec-precision", default="fp32", help="Codec ONNX export precision")
    parser.add_argument("--voices-only", action="store_true", help="Only pre-encode the 94 speaker references")
    parser.add_argument(
        "--stage",
        choices=["all", "voices-and-codec", "fast-ar", "slow-ar", "quantize"],
        default="all",
        help="Internal stage selector for memory-isolated subprocess execution",
    )
    parser.add_argument("--local-src", default=None, help="Resolved local snapshot path")
    args = parser.parse_args()

    if args.local_src:
        src_path = Path(args.local_src)
    else:
        src_path = Path(args.model_id)
        if not src_path.is_dir():
            print(f"Downloading {args.model_id} from Hugging Face...")
            local_dir = snapshot_download(repo_id=args.model_id, token=args.hf_token)
            src_path = Path(local_dir)

    patch_source_files(src_path)

    out_dir = Path(args.output_dir)
    voices_dir = Path(args.voices_dir)
    registry_path = Path(args.registry)

    if args.stage == "voices-and-codec":
        run_stage_voices_and_codec(src_path, out_dir, voices_dir, registry_path, args.model_id)
        return
    if args.stage == "fast-ar":
        run_stage_fast_ar(src_path, out_dir, args.model_id)
        return
    if args.stage == "slow-ar":
        run_stage_slow_ar(src_path, out_dir, args.model_id)
        return
    if args.stage == "quantize":
        run_stage_quantize(src_path, out_dir, args.model_id, args.precision)
        return

    # Orchestrate each stage in its own subprocess so the OS reclaims 100% of RAM between stages
    stages = ["voices-and-codec"] if args.voices_only else ["voices-and-codec", "fast-ar", "slow-ar", "quantize"]
    for stg in stages:
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            stg,
            "--model-id",
            args.model_id,
            "--local-src",
            str(src_path),
            "--output-dir",
            str(out_dir),
            "--voices-dir",
            str(voices_dir),
            "--registry",
            str(registry_path),
            "--precision",
            args.precision,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
