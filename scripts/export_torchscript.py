#!/usr/bin/env python3
"""
Standalone, memory-isolated TorchScript exporter for `phosai/realistic-v1` (Audio8-TTS 0.6B).

Produces self-contained PyTorch LibTorch (`tch-rs` 0.18 / Torch 2.5.1) artifacts in `--output-dir`:
  1. `slow_ar_fp16.pt` / `slow_ar_fp32.pt`     - Slow AR 0.6B transformer with in-place KV cache
  2. `fast_ar_fp16.pt` / `fast_ar_fp32.pt`     - Fused single-call Fast AR codebook loop + GPU sampler
  3. `codec_decoder_fp16.pt` / `codec_decoder_fp32.pt` - Streaming ArkttsCodec decoder
  4. `registration/codec_encoder_fp32.pt`      - Zero-shot reference audio encoder
  5. `runtime_manifest.json` + tokenizer files
  6. Pre-encoded 94 African speaker voice profiles in `--voices-dir/*.json`

Every traced ScriptModule has:
  - All weights/buffers converted to the target dtype BEFORE tracing (never post-trace `.half()`).
  - All plain tensor attributes registered as `nn.Module` buffers so `CModule::load_on_device`
    moves the entire module graph cleanly to `cuda:0` or `cpu`.
  - Any CPU `prim::Constant[Device="cpu"]` nodes rewritten to `prim::device(%input)` so models
    traced on a CPU build host execute natively on CUDA GPUs without device mismatch errors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "phosai/realistic-v1")
MAX_REF_AUDIO_SECONDS = 10.0


def sanitize_dirname(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def get_persistent_cache_dir(fingerprint: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / f"audio8_ts_cache_{fingerprint}_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_voice_cache_dirs(fingerprint: str) -> List[Path]:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return [
        hf_home / f"audio8_ts_cache_{fingerprint}_v1" / "voices",
        hf_home / f"audio8_build_cache_{fingerprint}_v5" / "voices",
        hf_home / f"audio8_build_cache_{fingerprint}_v2" / "voices",
        hf_home / f"audio8_build_cache_{fingerprint}" / "voices",
    ]


def patch_source_files(src_path: Path) -> None:
    """
    Patch downloaded HF modeling files so TorchScript tracing compiles cleanly:
      - avoid in-place bitwise ops (&=, |=, ^=)
      - replace complex torch.polar with real cos/sin math
      - keep _rope in real arithmetic without bfloat16 casts
      - crop ConvNeXt / ResidualUnit residual mismatches on arbitrary chunk lengths
      - keep `from __future__` imports at the top of the file
    """
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
                future_lines = [l for l in content.splitlines() if l.startswith("from __future__")]
                other_lines = [l for l in content.splitlines() if not l.startswith("from __future__")]
                content = "\n".join(future_lines + [""] + other_lines)
            if content != orig:
                print(f"Patched compatibility fixes in {py_file.name}.")
                py_file.write_text(content, encoding="utf-8")
        except Exception as exc:
            print(f"[Warn] Could not patch {py_file}: {exc}")


def prepare_module_for_torchscript(
    module: nn.Module,
    target_dtype: torch.dtype,
    keep_rope_float32: bool = True,
) -> nn.Module:
    """
    Fuse weight_norm/parametrizations, detach parameters/buffers, convert floating tensors to
    `target_dtype`, and promote plain `torch.Tensor` attributes into registered buffers so
    `torch::jit::Module::to(device)` moves every tensor in the graph to CUDA.
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

    module.eval().requires_grad_(False)
    for p in module.parameters():
        t = p.detach().to(target_dtype).clone() if p.is_floating_point() else p.detach().clone()
        t.requires_grad_(False)
        p.data = t
        p.requires_grad_(False)

    for name, b in list(module.named_buffers()):
        if not isinstance(b, torch.Tensor):
            continue
        is_rope = keep_rope_float32 and ("freqs_cis" in name)
        dt = torch.float32 if is_rope else target_dtype
        t = b.detach().to(dt).clone() if b.is_floating_point() else b.detach().clone()
        t.requires_grad_(False)
        b.data = t
        b.requires_grad_(False)

    for sub in module.modules():
        for k, v in list(sub.__dict__.items()):
            if isinstance(v, torch.Tensor) and not k.startswith("_"):
                is_rope = keep_rope_float32 and ("freqs_cis" in k)
                dt = torch.float32 if is_rope else target_dtype
                detached = v.detach().to(dt).clone() if v.is_floating_point() else v.detach().clone()
                detached.requires_grad_(False)
                del sub.__dict__[k]
                sub.register_buffer(k, detached)

    return module


def rewrite_cpu_device_constants(script_mod: torch.jit.ScriptModule) -> int:
    """
    Rewrite any `prim::Constant` with `Device` type (`cpu`) baked during CPU tracing
    into `prim::device(%anchor_tensor)` so the saved `.pt` runs on CUDA GPUs cleanly.
    """
    replaced_count = 0

    def patch_graph(root_g: torch._C.Graph, anchor_val: torch._C.Value) -> int:
        nonlocal replaced_count
        dev_node = None

        def get_dev_out() -> torch._C.Value:
            nonlocal dev_node
            if dev_node is None:
                dev_node = root_g.create("prim::device", [anchor_val])
                dev_node.output().setType(torch._C.DeviceObjType.get())
                nodes = list(root_g.nodes())
                if nodes:
                    dev_node.insertBefore(nodes[0])
                else:
                    root_g.appendNode(dev_node)
            return dev_node.output()

        def walk_block(blk: Any) -> None:
            nonlocal replaced_count
            for n in list(blk.nodes()):
                for inner in n.blocks():
                    walk_block(inner)
                if n.kind() == "prim::Constant" and str(n.output().type()) == "Device":
                    if "cpu" in str(n).lower():
                        n.output().replaceAllUsesWith(get_dev_out())
                        n.destroy()
                        replaced_count += 1

        walk_block(root_g)
        torch._C._jit_pass_lint(root_g)
        return replaced_count

    for sub in script_mod.modules():
        if hasattr(sub, "_c"):
            for method_name in ["forward"]:
                if sub._c._has_method(method_name):
                    g = sub._c._get_method(method_name).graph
                    anchor = next((inp for inp in g.inputs() if "Tensor" in str(inp.type())), None)
                    if anchor is not None:
                        patch_graph(g, anchor)
    return replaced_count


def assert_no_cpu_device_constants(script_mod: torch.jit.ScriptModule, label: str) -> None:
    graph_str = str(script_mod.inlined_graph)
    if 'prim::Constant[value="cpu"]' in graph_str:
        raise RuntimeError(
            f"Exported TorchScript module '{label}' still contains hardcoded CPU device constant!"
        )


def load_arktts_codec(src_path: Path, device: str = "cpu") -> nn.Module:
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
    return codec.eval().to(device=device, dtype=torch.float32)


@torch.jit.script
def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings (`freqs_cis` of shape `[L, head_dim // 2, 2]`)."""
    b = x.shape[0]
    s = x.shape[1]
    h = x.shape[2]
    d = x.shape[3]
    shaped = x.float().reshape((b, s, h, d // 2, 2))
    rope = freqs_cis.float()
    if rope.ndim == 2:
        rope = torch.stack((torch.cos(rope), torch.sin(rope)), dim=-1)
    rope = rope.unsqueeze(0).unsqueeze(2)
    out_0 = shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1]
    out_1 = shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1]
    return torch.stack((out_0, out_1), dim=-1).reshape((b, s, h, d)).to(x.dtype)


@torch.jit.script
def slice_and_mask_slow(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    input_pos: torch.Tensor,
    valid_len: torch.Tensor,
    repeat_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vlen = int(valid_len[0].item())
    k_slice = k_cache[:, :, :vlen, :]
    v_slice = v_cache[:, :, :vlen, :]
    if repeat_factor > 1:
        k_rep = k_slice.repeat_interleave(repeat_factor, dim=1)
        v_rep = v_slice.repeat_interleave(repeat_factor, dim=1)
    else:
        k_rep = k_slice
        v_rep = v_slice
    k_pos = torch.arange(vlen, dtype=torch.int64, device=input_pos.device)
    mask = (k_pos.unsqueeze(0) <= input_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return k_rep, v_rep, mask


@torch.jit.script
def slice_and_mask_fast(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    input_pos: torch.Tensor,
    repeat_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_cb = k_cache.shape[2]
    if repeat_factor > 1:
        k_rep = k_cache.repeat_interleave(repeat_factor, dim=1)
        v_rep = v_cache.repeat_interleave(repeat_factor, dim=1)
    else:
        k_rep = k_cache
        v_rep = v_cache
    k_pos = torch.arange(max_cb, dtype=torch.int64, device=input_pos.device)
    mask = (k_pos <= input_pos[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return k_rep, v_rep, mask


@torch.jit.script
def sample_fast_gpu(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    """
    GPU-resident Top-K / Top-P / Temperature sampler returning `[1]` `int64` token ID.
    """
    t = float(temperature[0].item())
    k_raw = int(top_k[0].item())
    p_raw = float(top_p[0].item())
    logits_f = logits.float()

    if t <= 0.02 or k_raw == 1:
        return torch.argmax(logits_f, dim=-1, keepdim=True).to(torch.int64)

    scaled = logits_f / max(t, 1e-5)
    vocab = scaled.shape[0]
    k_limit = vocab if k_raw <= 0 or k_raw >= vocab else k_raw

    if k_limit < vocab:
        top_vals, top_idx = torch.topk(scaled, k_limit)
        probs_k = torch.softmax(top_vals, dim=-1)
        if 0.0 < p_raw < 1.0:
            cumsum = torch.cumsum(probs_k, dim=-1)
            remove = (cumsum - probs_k) > p_raw
            remove[0] = False
            probs_k = torch.where(remove, torch.zeros_like(probs_k), probs_k)
        probs_k = probs_k / probs_k.sum().clamp_min(1e-9)
        picked = torch.multinomial(probs_k, 1)
        return top_idx.index_select(0, picked).to(torch.int64)

    probs = torch.softmax(scaled, dim=-1)
    if 0.0 < p_raw < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=-1)
        remove = (cumsum - sorted_p) > p_raw
        remove[0] = False
        sorted_p = torch.where(remove, torch.zeros_like(sorted_p), sorted_p)
        probs = torch.zeros_like(probs).scatter_(0, sorted_i, sorted_p)
    probs = probs / probs.sum().clamp_min(1e-9)
    return torch.multinomial(probs, 1).to(torch.int64)


class SlowARTorchWrapper(nn.Module):
    """
    TorchScript wrapper for Audio8 0.6B Slow AR transformer.

    Supports both multi-token prefill (`seq_len >= 1`) and single-token decode (`seq_len == 1`)
    using pre-allocated GPU-resident KV caches `[num_layers, 1, n_local_heads, max_seq_len, head_dim]`,
    updated in-place via `index_copy_`, and sliced dynamically to `valid_len`.

    Returns:
      - `logits_last`: `[codebook_size + 1]` in `float32` (semantic tokens + EOS for Rust RAS/stop check)
      - `fast_hidden_last`: `[1, 1, dim]` in `target_dtype` (kept on GPU for FastARLoop)
    """

    def __init__(self, model: nn.Module, target_dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
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
        self.scale_codebook_embeddings = bool(getattr(cfg, "scale_codebook_embeddings", False))
        self.emb_scale = 1.0 / math.sqrt(self.num_codebooks + 1)
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_local_heads * self.head_dim
        self.repeat_factor = self.n_head // self.n_local_heads

        self.emb_layer = getattr(model, "embeddings", None) or getattr(model, "embed_tokens")
        self.codebook_embeddings = model.codebook_embeddings
        self.layers = model.layers
        self.norm = model.norm

        offsets = torch.arange(self.num_codebooks, dtype=torch.long) * self.codebook_size
        self.register_buffer("codebook_offsets", offsets.view(1, self.num_codebooks, 1))
        self.register_buffer("freqs_cis", model.freqs_cis.detach().float().clone())

        head_layer = (
            getattr(model, "output", None)
            or getattr(model, "semantic_output", None)
            or getattr(model, "lm_head", None)
            or self.emb_layer
        )
        head_weight = head_layer.weight.detach()
        if head_weight.shape[0] == self.codebook_size + 1:
            proj_w = head_weight.to(target_dtype).clone()
            head_bias = getattr(head_layer, "bias", None)
            proj_b = head_bias.detach().to(target_dtype).clone() if head_bias is not None else None
        else:
            sem_w = head_weight[self.semantic_begin_id : self.semantic_end_id + 1]
            eos_w = head_weight[self.im_end_id : self.im_end_id + 1]
            proj_w = torch.cat([sem_w, eos_w], dim=0).to(target_dtype).clone()
            proj_b = None

        self.register_buffer("slow_head_weight", proj_w)
        if proj_b is not None:
            self.register_buffer("slow_head_bias", proj_b)
            self.has_head_bias = True
        else:
            self.register_buffer("slow_head_bias", torch.zeros(1, dtype=target_dtype))
            self.has_head_bias = False

    def forward(
        self,
        codes: torch.Tensor,
        input_pos: torch.Tensor,
        valid_len: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row0 = codes[:, 0, :]
        vq_rows = codes[:, 1:, :]

        text_emb = self.emb_layer(row0)
        offset_vq = vq_rows + self.codebook_offsets
        cb_emb = self.codebook_embeddings(offset_vq).sum(dim=1)

        if self.scale_codebook_embeddings:
            cb_emb = cb_emb * self.emb_scale
            text_emb_scaled = text_emb * self.emb_scale
        else:
            text_emb_scaled = text_emb

        is_semantic = (row0 >= self.semantic_begin_id) & (row0 <= self.semantic_end_id)
        hidden = torch.where(is_semantic.unsqueeze(-1), text_emb_scaled + cb_emb, text_emb)
        rope_cis = self.freqs_cis[input_pos]

        for idx, layer in enumerate(self.layers):
            normed = layer.attention_norm(hidden)
            attn = layer.attention
            q, k, v = attn.wqkv(normed).split((self.q_size, self.kv_size, self.kv_size), dim=-1)
            q = q.view(1, -1, self.n_head, self.head_dim)
            k = k.view(1, -1, self.n_local_heads, self.head_dim)
            v = v.view(1, -1, self.n_local_heads, self.head_dim)

            if getattr(attn, "qk_norm", False) and hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            q = apply_rope(q, rope_cis).transpose(1, 2)
            k = apply_rope(k, rope_cis).transpose(1, 2)
            v = v.transpose(1, 2)

            k_caches[idx].index_copy_(2, input_pos, k)
            v_caches[idx].index_copy_(2, input_pos, v)

            k_rep, v_rep, mask = slice_and_mask_slow(
                k_caches[idx], v_caches[idx], input_pos, valid_len, self.repeat_factor
            )
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=mask, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, -1, self.q_size)

            hidden = hidden + attn.wo(attn_out)
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

        hidden_last = hidden[:, -1:, :]
        normed_last = self.norm(hidden_last)
        fast_hidden_last = normed_last if self.norm_fastlayer_input else hidden_last

        if self.has_head_bias:
            logits = F.linear(normed_last, self.slow_head_weight, self.slow_head_bias)
        else:
            logits = F.linear(normed_last, self.slow_head_weight)

        return logits.float().squeeze(0).squeeze(0), fast_hidden_last


class FastARCoreStepWrapper(nn.Module):
    """Single-step Fast AR 4-layer block with in-place KV cache update, trace-friendly."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        cfg = model.config
        self.num_fast_layers = int(cfg.n_fast_layer)
        self.n_head = int(cfg.fast_n_head)
        self.n_local_heads = int(cfg.fast_n_local_heads)
        self.head_dim = int(cfg.fast_head_dim)
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_local_heads * self.head_dim
        self.repeat_factor = self.n_head // self.n_local_heads

        self.fast_layers = model.fast_layers
        self.fast_norm = model.fast_norm
        self.fast_output = model.fast_output
        self.register_buffer("fast_freqs_cis", model.fast_freqs_cis.detach().float().clone())

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
    ) -> torch.Tensor:
        rope_cis = self.fast_freqs_cis[input_pos]
        for idx, layer in enumerate(self.fast_layers):
            normed = layer.attention_norm(hidden)
            attn = layer.attention
            q, k, v = attn.wqkv(normed).split((self.q_size, self.kv_size, self.kv_size), dim=-1)
            q = q.view(1, 1, self.n_head, self.head_dim)
            k = k.view(1, 1, self.n_local_heads, self.head_dim)
            v = v.view(1, 1, self.n_local_heads, self.head_dim)

            if getattr(attn, "qk_norm", False) and hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            q = apply_rope(q, rope_cis).transpose(1, 2)
            k = apply_rope(k, rope_cis).transpose(1, 2)
            v = v.transpose(1, 2)

            k_caches[idx].index_copy_(2, input_pos, k)
            v_caches[idx].index_copy_(2, input_pos, v)

            k_rep, v_rep, mask = slice_and_mask_fast(
                k_caches[idx], v_caches[idx], input_pos, self.repeat_factor
            )
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=mask, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, self.q_size)

            hidden = hidden + attn.wo(attn_out)
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

        return self.fast_output(self.fast_norm(hidden))


class FastARLoopTorchWrapper(nn.Module):
    """
    Fused single-call Fast AR generator running all codebooks (`pos=0..num_codebooks-1`)
    plus GPU Top-K/Top-P/Temperature sampling in one LibTorch call per frame.

    Inputs:
      - `slow_hidden`: `[1, 1, dim]` (`target_dtype`)
      - `first_code`: `[1]` (`int64`)
      - `temperature`: `[1]` (`float32`)
      - `top_p`: `[1]` (`float32`)
      - `top_k`: `[1]` (`int64`)
    Returns:
      - `codes`: `[num_codebooks]` (`int64`)
    """

    def __init__(
        self,
        step_traced: torch.jit.ScriptModule,
        fast_project_in: nn.Module,
        fast_embeddings: nn.Module,
        num_fast_layers: int,
        fast_n_local_heads: int,
        fast_head_dim: int,
        num_codebooks: int,
    ) -> None:
        super().__init__()
        self.step_mod = step_traced
        self.fast_project_in = fast_project_in
        self.fast_embeddings = fast_embeddings
        self.num_fast_layers = num_fast_layers
        self.fast_n_local_heads = fast_n_local_heads
        self.fast_head_dim = fast_head_dim
        self.num_codebooks = num_codebooks

    def forward(
        self,
        slow_hidden: torch.Tensor,
        first_code: torch.Tensor,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        top_k: torch.Tensor,
    ) -> torch.Tensor:
        dev = slow_hidden.device
        dt = slow_hidden.dtype
        kc = torch.zeros(
            (self.num_fast_layers, 1, self.fast_n_local_heads, self.num_codebooks, self.fast_head_dim),
            dtype=dt,
            device=dev,
        )
        vc = torch.zeros(
            (self.num_fast_layers, 1, self.fast_n_local_heads, self.num_codebooks, self.fast_head_dim),
            dtype=dt,
            device=dev,
        )
        out = torch.zeros((self.num_codebooks,), dtype=torch.int64, device=dev)
        out[0] = first_code[0]

        pos_seq = torch.arange(self.num_codebooks, dtype=torch.int64, device=dev)
        h0 = self.fast_project_in(slow_hidden)
        _ = self.step_mod(h0, pos_seq[0:1], kc, vc)

        prev = first_code.view(1, 1)
        for idx in range(1, self.num_codebooks):
            emb = self.fast_embeddings(prev)
            logits = self.step_mod(emb, pos_seq[idx : idx + 1], kc, vc)[0, 0]
            nxt = sample_fast_gpu(logits, temperature, top_p, top_k)
            out[idx] = nxt[0]
            prev = nxt.view(1, 1)

        return out


class CodecDecoderTorchWrapper(nn.Module):
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
        return out.float()


@torch.jit.script
def _waveform_length_tensor(wav_2d: torch.Tensor) -> torch.Tensor:
    return torch.full((1,), wav_2d.shape[-1], dtype=torch.int64, device=wav_2d.device)


class CodecEncoderTorchWrapper(nn.Module):
    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        wav_2d = audio.squeeze(1) if audio.ndim == 3 else audio
        lengths = _waveform_length_tensor(wav_2d)
        encode_method = getattr(self.codec, "encode", None)
        if hasattr(encode_method, "__wrapped__"):
            res = encode_method.__wrapped__(self.codec, wav_2d, lengths)
        else:
            res = self.codec.encode(wav_2d, lengths)
        indices = res[0] if isinstance(res, tuple) else res
        return indices.to(torch.int64)


def pre_encode_speaker_references(
    codec: nn.Module,
    registry_path: Path,
    voices_dir: Path,
    sample_rate: int,
    hop_length: int,
    model_fingerprint: str,
    device: str,
) -> None:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not registry_path.is_file():
        print(f"No speaker_registry.json found at {registry_path}, skipping pre-encoding.")
        return

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    voices_dir.mkdir(parents=True, exist_ok=True)
    primary_cache = get_persistent_cache_dir(model_fingerprint) / "voices"
    primary_cache.mkdir(parents=True, exist_ok=True)
    legacy_caches = get_voice_cache_dirs(model_fingerprint)

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

        cached_json = primary_cache / f"{safe_stem}.json"
        cached_codes = primary_cache / safe_name / "codes.npy"
        cached_meta = primary_cache / safe_name / "meta.json"

        if not cached_json.is_file():
            for cdir in legacy_caches:
                if cdir.is_dir():
                    c_json = cdir / f"{safe_stem}.json"
                    c_c = cdir / safe_name / "codes.npy"
                    c_m = cdir / safe_name / "meta.json"
                    if c_json.is_file() and c_c.is_file() and c_m.is_file():
                        cached_codes.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(c_json, cached_json)
                        shutil.copy2(c_c, cached_codes)
                        shutil.copy2(c_m, cached_meta)
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


def validate_codec_decoder_pt(pt_path: Path, num_cb: int) -> None:
    loaded = torch.jit.load(str(pt_path), map_location="cpu").eval()
    with torch.no_grad():
        for test_len in [2, 4, 8, 11, 23]:
            dummy = torch.zeros((1, num_cb, test_len), dtype=torch.long)
            out = loaded(dummy)
            if out.dtype != torch.float32:
                raise RuntimeError(f"Expected float32 audio from {pt_path.name}, got {out.dtype}")
            print(f"  Validated {pt_path.name} len={test_len} -> {tuple(out.shape)} ({out.dtype})")


def run_stage_voices_and_codec(
    src_path: Path,
    out_dir: Path,
    voices_dir: Path,
    registry_path: Path,
    model_id: str,
    precisions: List[str],
) -> None:
    print("=== [Stage 1/4] Exporting Codec TorchScript & pre-encoding 94 speaker voices ===")
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

    model_cache_dir = get_persistent_cache_dir(fingerprint) / "model"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Always prepare FP32 codec (used for speaker encoding + safe CPU/fallback decoding)
    codec_fp32 = load_arktts_codec(src_path, device="cpu")
    prepare_module_for_torchscript(codec_fp32, torch.float32)

    dec_fp32_path = out_dir / "codec_decoder_fp32.pt"
    cached_dec_fp32 = model_cache_dir / "codec_decoder_fp32.pt"
    if cached_dec_fp32.is_file():
        print(f"Restoring {dec_fp32_path.name} from cache...")
        shutil.copy2(cached_dec_fp32, dec_fp32_path)
    else:
        dec_wrapper = CodecDecoderTorchWrapper(codec_fp32).eval()
        dummy_vq = torch.zeros((1, num_cb, 8), dtype=torch.long)
        with torch.no_grad():
            traced = torch.jit.trace(dec_wrapper, (dummy_vq,))
        rewrite_cpu_device_constants(traced)
        assert_no_cpu_device_constants(traced, "codec_decoder_fp32")
        torch.jit.save(traced, str(dec_fp32_path))
        validate_codec_decoder_pt(dec_fp32_path, num_cb)
        shutil.copy2(dec_fp32_path, cached_dec_fp32)
        del traced, dec_wrapper

    reg_dir = out_dir / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32_path = reg_dir / "codec_encoder_fp32.pt"
    cached_enc = model_cache_dir / "codec_encoder_fp32.pt"
    if cached_enc.is_file():
        print(f"Restoring {enc_fp32_path.name} from cache...")
        shutil.copy2(cached_enc, enc_fp32_path)
    else:
        try:
            enc_wrapper = CodecEncoderTorchWrapper(codec_fp32).eval()
            dummy_wav = torch.zeros((1, 1, hop_length * 8), dtype=torch.float32)
            with torch.no_grad():
                traced_enc = torch.jit.trace(enc_wrapper, (dummy_wav,))
            rewrite_cpu_device_constants(traced_enc)
            assert_no_cpu_device_constants(traced_enc, "codec_encoder_fp32")
            torch.jit.save(traced_enc, str(enc_fp32_path))
            # Verify dynamic length
            loaded_enc = torch.jit.load(str(enc_fp32_path), map_location="cpu").eval()
            with torch.no_grad():
                res = loaded_enc(torch.zeros((1, 1, hop_length * 12), dtype=torch.float32))
                print(f"  Validated {enc_fp32_path.name} -> {tuple(res.shape)} ({res.dtype})")
            shutil.copy2(enc_fp32_path, cached_enc)
            del traced_enc, enc_wrapper, loaded_enc
        except Exception as exc:
            print(f"[Warn] Optional codec_encoder_fp32.pt export skipped ({exc}); pre-encoded voices still work.")

    # Pre-encode voices using the FP32 codec BEFORE converting/unloading
    pre_device = "cuda" if torch.cuda.is_available() else "cpu"
    if pre_device != "cpu":
        codec_fp32 = codec_fp32.to(device=pre_device, dtype=torch.float32)
    pre_encode_speaker_references(
        codec=codec_fp32,
        registry_path=registry_path,
        voices_dir=voices_dir,
        sample_rate=sample_rate,
        hop_length=hop_length,
        model_fingerprint=fingerprint,
        device=pre_device,
    )
    del codec_fp32
    gc.collect()

    # Export Codec Decoder FP16 if requested
    if "fp16" in precisions:
        dec_fp16_path = out_dir / "codec_decoder_fp16.pt"
        cached_dec_fp16 = model_cache_dir / "codec_decoder_fp16.pt"
        if cached_dec_fp16.is_file():
            print(f"Restoring {dec_fp16_path.name} from cache...")
            shutil.copy2(cached_dec_fp16, dec_fp16_path)
        else:
            try:
                codec_fp16 = load_arktts_codec(src_path, device="cpu")
                prepare_module_for_torchscript(codec_fp16, torch.float16)
                dec_wrapper_16 = CodecDecoderTorchWrapper(codec_fp16).eval()
                dummy_vq = torch.zeros((1, num_cb, 8), dtype=torch.long)
                with torch.no_grad():
                    traced_16 = torch.jit.trace(dec_wrapper_16, (dummy_vq,))
                rewrite_cpu_device_constants(traced_16)
                assert_no_cpu_device_constants(traced_16, "codec_decoder_fp16")
                torch.jit.save(traced_16, str(dec_fp16_path))
                validate_codec_decoder_pt(dec_fp16_path, num_cb)
                shutil.copy2(dec_fp16_path, cached_dec_fp16)
                del traced_16, dec_wrapper_16, codec_fp16
            except Exception as exc:
                if dec_fp16_path.is_file():
                    dec_fp16_path.unlink()
                print(
                    f"[Warn] Codec Decoder FP16 trace unsupported on CPU host ({exc}); "
                    "will keep Codec Decoder in FP32."
                )
            gc.collect()


def export_fast_ar_for_dtype(
    src_path: Path,
    out_path: Path,
    cache_path: Path,
    target_dtype: torch.dtype,
    label: str,
) -> None:
    from transformers import AutoModel

    if cache_path.is_file():
        print(f"Restoring {out_path.name} from cache...")
        shutil.copy2(cache_path, out_path)
        return

    model = AutoModel.from_pretrained(
        str(src_path),
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    cfg = model.config

    for attr in ["layers", "embeddings", "embed_tokens", "codebook_embeddings", "output", "semantic_output", "lm_head", "norm"]:
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()

    prepare_module_for_torchscript(model, target_dtype)
    num_fast = int(cfg.n_fast_layer)
    fast_local_heads = int(cfg.fast_n_local_heads)
    fast_head_dim = int(cfg.fast_head_dim)
    num_cb = int(cfg.num_codebooks)
    dim = int(cfg.dim)

    step_wrapper = FastARCoreStepWrapper(model).eval()
    with torch.no_grad():
        dummy_slow_hidden = torch.zeros((1, 1, dim), dtype=target_dtype)
        dummy_fast_emb = model.fast_project_in(dummy_slow_hidden)
        dummy_pos = torch.zeros((1,), dtype=torch.long)
        dummy_k = torch.zeros((num_fast, 1, fast_local_heads, num_cb, fast_head_dim), dtype=target_dtype)
        dummy_v = torch.zeros((num_fast, 1, fast_local_heads, num_cb, fast_head_dim), dtype=target_dtype)
        step_traced = torch.jit.trace(step_wrapper, (dummy_fast_emb, dummy_pos, dummy_k, dummy_v))

    rewrite_cpu_device_constants(step_traced)

    loop_wrapper = FastARLoopTorchWrapper(
        step_traced=step_traced,
        fast_project_in=model.fast_project_in,
        fast_embeddings=model.fast_embeddings,
        num_fast_layers=num_fast,
        fast_n_local_heads=fast_local_heads,
        fast_head_dim=fast_head_dim,
        num_codebooks=num_cb,
    ).eval()
    scripted_loop = torch.jit.script(loop_wrapper)
    rewrite_cpu_device_constants(scripted_loop)
    assert_no_cpu_device_constants(scripted_loop, label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted_loop, str(out_path))

    # Validate load + execution
    loaded = torch.jit.load(str(out_path), map_location="cpu").eval()
    with torch.no_grad():
        out_codes = loaded(
            torch.randn((1, 1, dim), dtype=target_dtype),
            torch.tensor([7], dtype=torch.long),
            torch.tensor([0.7], dtype=torch.float32),
            torch.tensor([0.8], dtype=torch.float32),
            torch.tensor([20], dtype=torch.long),
        )
        if tuple(out_codes.shape) != (num_cb,) or out_codes.dtype != torch.int64:
            raise RuntimeError(f"Unexpected FastARLoop output {out_codes.shape} {out_codes.dtype}")
        print(f"  Validated {out_path.name} -> {out_codes.tolist()}")

    shutil.copy2(out_path, cache_path)
    del model, step_wrapper, step_traced, loop_wrapper, scripted_loop, loaded
    gc.collect()


def run_stage_fast_ar(
    src_path: Path,
    out_dir: Path,
    model_id: str,
    precisions: List[str],
) -> None:
    print(f"=== [Stage 2/4] Exporting Fast AR TorchScript ({','.join(precisions)}) ===")
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]
    model_cache_dir = get_persistent_cache_dir(fingerprint) / "model"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    for prec in precisions:
        dt = dtype_map[prec]
        fname = f"fast_ar_{prec}.pt"
        export_fast_ar_for_dtype(src_path, out_dir / fname, model_cache_dir / fname, dt, fname)


def export_slow_ar_for_dtype(
    src_path: Path,
    out_path: Path,
    cache_path: Path,
    target_dtype: torch.dtype,
    label: str,
) -> None:
    from transformers import AutoModel

    if cache_path.is_file():
        print(f"Restoring {out_path.name} from cache...")
        shutil.copy2(cache_path, out_path)
        return

    model = AutoModel.from_pretrained(
        str(src_path),
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    cfg = model.config

    for attr in ["fast_layers", "fast_embeddings", "fast_output", "fast_project_in", "fast_norm"]:
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()

    prepare_module_for_torchscript(model, target_dtype)
    slow_wrapper = SlowARTorchWrapper(model, target_dtype).eval()

    num_layers = int(cfg.n_layer)
    n_local_heads = int(cfg.n_local_heads)
    head_dim = int(cfg.head_dim)
    num_cb = int(cfg.num_codebooks)
    dim = int(cfg.dim)
    cb_size = int(cfg.codebook_size)

    with torch.no_grad():
        dummy_codes = torch.zeros((1, num_cb + 1, 5), dtype=torch.long)
        dummy_pos = torch.arange(5, dtype=torch.long)
        dummy_valid = torch.tensor([5], dtype=torch.long)
        dummy_kc = torch.zeros((num_layers, 1, n_local_heads, 32, head_dim), dtype=target_dtype)
        dummy_vc = torch.zeros((num_layers, 1, n_local_heads, 32, head_dim), dtype=target_dtype)
        traced = torch.jit.trace(
            slow_wrapper, (dummy_codes, dummy_pos, dummy_valid, dummy_kc, dummy_vc)
        )

    rewrite_cpu_device_constants(traced)
    assert_no_cpu_device_constants(traced, label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out_path))

    # Validate reload with BOTH prefill (seq_len=9) and decode (seq_len=1)
    loaded = torch.jit.load(str(out_path), map_location="cpu").eval()
    with torch.no_grad():
        kc_test = torch.zeros((num_layers, 1, n_local_heads, 32, head_dim), dtype=target_dtype)
        vc_test = torch.zeros((num_layers, 1, n_local_heads, 32, head_dim), dtype=target_dtype)
        l_prefill, h_prefill = loaded(
            torch.zeros((1, num_cb + 1, 9), dtype=torch.long),
            torch.arange(9, dtype=torch.long),
            torch.tensor([9], dtype=torch.long),
            kc_test,
            vc_test,
        )
        l_step, h_step = loaded(
            torch.zeros((1, num_cb + 1, 1), dtype=torch.long),
            torch.tensor([9], dtype=torch.long),
            torch.tensor([10], dtype=torch.long),
            kc_test,
            vc_test,
        )
        if tuple(l_prefill.shape) != (cb_size + 1,) or l_prefill.dtype != torch.float32:
            raise RuntimeError(f"Invalid SlowAR prefill logits {l_prefill.shape} {l_prefill.dtype}")
        if tuple(h_step.shape) != (1, 1, dim) or h_step.dtype != target_dtype:
            raise RuntimeError(f"Invalid SlowAR hidden {h_step.shape} {h_step.dtype}")
        print(f"  Validated {out_path.name} prefill + decode -> logits={tuple(l_step.shape)} hidden={tuple(h_step.shape)} ({h_step.dtype})")

    shutil.copy2(out_path, cache_path)
    del model, slow_wrapper, traced, loaded
    gc.collect()


def run_stage_slow_ar(
    src_path: Path,
    out_dir: Path,
    model_id: str,
    precisions: List[str],
) -> None:
    print(f"=== [Stage 3/4] Exporting Slow AR TorchScript ({','.join(precisions)}) ===")
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]
    model_cache_dir = get_persistent_cache_dir(fingerprint) / "model"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    for prec in precisions:
        dt = dtype_map[prec]
        fname = f"slow_ar_{prec}.pt"
        export_slow_ar_for_dtype(src_path, out_dir / fname, model_cache_dir / fname, dt, fname)


def run_stage_manifest(
    src_path: Path,
    out_dir: Path,
    model_id: str,
    precisions: List[str],
) -> None:
    from transformers import AutoConfig

    print(f"=== [Stage 4/4] Writing runtime_manifest.json ({','.join(precisions)}) ===")
    cfg = AutoConfig.from_pretrained(str(src_path), trust_remote_code=True)
    fingerprint = hashlib.sha256(str(model_id).encode()).hexdigest()[:16]

    slow_models: Dict[str, str] = {}
    fast_models: Dict[str, str] = {}
    codec_models: Dict[str, str] = {}
    available_precisions: List[str] = []
    available_codec_precisions: List[str] = []

    for prec in ["fp16", "fp32"]:
        if (out_dir / f"slow_ar_{prec}.pt").is_file():
            slow_models[prec] = f"slow_ar_{prec}.pt"
            available_precisions.append(prec)
        if (out_dir / f"fast_ar_{prec}.pt").is_file():
            fast_models[prec] = f"fast_ar_{prec}.pt"
        if (out_dir / f"codec_decoder_{prec}.pt").is_file():
            codec_models[prec] = f"codec_decoder_{prec}.pt"
            available_codec_precisions.append(prec)

    default_prec = "fp16" if "fp16" in available_precisions else (available_precisions[0] if available_precisions else "fp32")
    default_codec_prec = "fp16" if "fp16" in available_codec_precisions else "fp32"

    manifest = {
        "model_id": model_id,
        "model_fingerprint": fingerprint,
        "default_precision": default_prec,
        "available_precisions": available_precisions,
        "default_codec_precision": default_codec_prec,
        "available_codec_precisions": available_codec_precisions,
        "slow_decode_models": slow_models,
        "fast_models": fast_models,
        "codec_models": codec_models,
        "codec_encoder_model": "registration/codec_encoder_fp32.pt",
        "sample_rate": int(getattr(cfg, "codec_sample_rate", getattr(cfg, "sample_rate", 44100))),
        "num_codebooks": int(cfg.num_codebooks),
        "codebook_size": int(cfg.codebook_size),
        "codec_hop_length": int(getattr(cfg, "codec_hop_length", 2048)),
        "stream_context_frames": 48,
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


def parse_precisions(raw: str) -> List[str]:
    vals = [p.strip().lower() for p in raw.split(",") if p.strip()]
    allowed = {"fp16", "fp32"}
    out: List[str] = []
    for v in vals:
        if v == "all":
            return ["fp16", "fp32"]
        if v not in allowed:
            raise ValueError(f"Unsupported precision '{v}'; choose fp16, fp32, or fp16,fp32")
        if v not in out:
            out.append(v)
    return out or ["fp16", "fp32"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export phosai/realistic-v1 to LibTorch TorchScript")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face repo ID or local path")
    parser.add_argument("--hf-token", default=DEFAULT_HF_TOKEN, help="Hugging Face access token")
    parser.add_argument("--output-dir", default="model", help="Target directory for .pt modules")
    parser.add_argument("--voices-dir", default="voices", help="Target directory for pre-encoded voices")
    parser.add_argument("--registry", default="speaker_registry.json", help="Path to speaker_registry.json")
    parser.add_argument(
        "--precision",
        default="fp16,fp32",
        help="Comma-separated export precisions (fp16, fp32, or fp16,fp32)",
    )
    parser.add_argument("--voices-only", action="store_true", help="Only pre-encode the 94 speaker references")
    parser.add_argument(
        "--stage",
        choices=["all", "voices-and-codec", "fast-ar", "slow-ar", "manifest"],
        default="all",
        help="Internal stage selector for memory-isolated subprocess execution",
    )
    parser.add_argument("--local-src", default=None, help="Resolved local snapshot path")
    args = parser.parse_args()

    precisions = parse_precisions(args.precision)

    if args.local_src:
        src_path = Path(args.local_src)
    else:
        src_path = Path(args.model_id)
        if not src_path.is_dir():
            from huggingface_hub import snapshot_download

            print(f"Downloading {args.model_id} from Hugging Face...")
            local_dir = snapshot_download(repo_id=args.model_id, token=args.hf_token)
            src_path = Path(local_dir)

    patch_source_files(src_path)

    out_dir = Path(args.output_dir)
    voices_dir = Path(args.voices_dir)
    registry_path = Path(args.registry)

    if args.stage == "voices-and-codec":
        run_stage_voices_and_codec(src_path, out_dir, voices_dir, registry_path, args.model_id, precisions)
        return
    if args.stage == "fast-ar":
        run_stage_fast_ar(src_path, out_dir, args.model_id, precisions)
        return
    if args.stage == "slow-ar":
        run_stage_slow_ar(src_path, out_dir, args.model_id, precisions)
        return
    if args.stage == "manifest":
        run_stage_manifest(src_path, out_dir, args.model_id, precisions)
        return

    stages = ["voices-and-codec"] if args.voices_only else ["voices-and-codec", "fast-ar", "slow-ar", "manifest"]
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
            ",".join(precisions),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
