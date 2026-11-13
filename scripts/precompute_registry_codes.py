#!/usr/bin/env python3
"""
Precomputes [num_codebooks, T] VQ codes for all speaker reference WAVs in
speaker_registry.json using ArkttsCodec, and updates speaker_registry.json in place.
"""

import gc
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def load_codec(model_path: Path, device: str = "cpu"):
    import importlib.util

    module_path = model_path / "modeling_arktts_codec.py"
    spec = importlib.util.spec_from_file_location("_audio8_codec", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import codec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with open(model_path / "config.json", encoding="utf-8") as f:
        from types import SimpleNamespace
        cfg = SimpleNamespace(**json.load(f))

    codec = module.ArkttsCodec(cfg)
    pth_path = model_path / "codec.pth"
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("generator.", ""): v for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True, assign=True)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return codec.eval().to(device=device, dtype=dtype), cfg


def main():
    root = Path(__file__).resolve().parent.parent
    registry_path = Path(os.getenv("REGISTRY_PATH", str(root / "speaker_registry.json")))
    model_path = Path(os.getenv("MODEL_PATH", "/models/phosai_realistic-v1"))
    if not model_path.exists():
        model_path = Path(os.getenv("MODEL_PATH", "/models/Audio8-TTS-Preview-0.6b"))
    if not model_path.exists():
        model_path = root / "model"

    if not registry_path.is_file():
        print(f"Registry not found at {registry_path}", file=sys.stderr)
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading ArkttsCodec from {model_path} on {device}...")
    codec, cfg = load_codec(model_path, device)
    sample_rate = int(getattr(cfg, "sample_rate", 24000))
    hop_length = int(getattr(cfg, "codec_frame_size", 2048))

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    total = len(registry)
    updated = 0
    print(f"Processing {total} speaker voices from {registry_path}...")

    for idx, (voice_id, entry) in enumerate(registry.items(), 1):
        if "vq_codes" in entry and entry["vq_codes"]:
            continue

        wav_rel = entry.get("reference_audio") or entry.get("audio_path")
        if not wav_rel:
            continue
        wav_path = root / wav_rel
        if not wav_path.is_file():
            wav_path = Path(wav_rel)
        if not wav_path.is_file():
            print(f"  [{idx}/{total}] Warning: audio not found: {wav_rel}")
            continue

        audio_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        mono = audio_np.mean(axis=1)
        if int(sr) != sample_rate:
            g = math.gcd(int(sr), sample_rate)
            mono = resample_poly(mono, sample_rate // g, int(sr) // g).astype(np.float32)

        # Pad to multiple of hop_length
        remainder = mono.size % hop_length
        if remainder:
            mono = np.pad(mono, (0, hop_length - remainder))

        with torch.no_grad():
            wav_t = torch.from_numpy(mono).unsqueeze(0).to(device=device, dtype=torch.float32)
            lengths = torch.tensor([mono.size], device=device, dtype=torch.long)
            indices, feature_lens = codec.encode(wav_t, lengths)
            n_frames = int(feature_lens[0].item())
            codes_np = indices[0, :, :n_frames].detach().cpu().numpy().astype(int)

        entry["vq_codes"] = codes_np.tolist()
        updated += 1
        if idx % 10 == 0 or idx == total:
            gc.collect()
            print(f"  [{idx}/{total}] Encoded {voice_id} (shape={codes_np.shape})...")

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Done! Updated {updated} voices with precomputed vq_codes in {registry_path}.")


if __name__ == "__main__":
    main()
