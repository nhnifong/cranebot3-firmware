#!/usr/bin/env python

"""Prepare a local copy of a SmolVLA checkpoint with a larger max_state_dim.

SmolVLA's `state_proj` layer has shape (hidden_size, max_state_dim). If your
dataset's observation.state has more dims than the checkpoint's
max_state_dim (32 for lerobot/smolvla_base), loading the pretrained weights
fails with a size mismatch.

This script downloads the checkpoint, zero-pads the extra `state_proj.weight`
input columns, and writes a local directory you can pass to lerobot-train via
`--policy.path=<output_dir>`. Zero-padding preserves the pretrained behavior
at initialization (the new state dims contribute nothing until finetuning
adapts them).

Usage:
    python src/nf_robot/ml/lerobot_expand_smolvla_state_dim.py \
        --source lerobot/smolvla_base \
        --output_dir models/smolvla_base_state64 \
        --max_state_dim 64
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand max_state_dim of a SmolVLA checkpoint.")
    parser.add_argument("--source", default="lerobot/smolvla_base", help="Source policy repo id or path")
    parser.add_argument("--output_dir", required=True, help="Where to write the expanded checkpoint")
    parser.add_argument("--max_state_dim", type=int, required=True, help="New max_state_dim (must be >= current)")
    args = parser.parse_args()

    src_dir = Path(snapshot_download(args.source))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((src_dir / "config.json").read_text())
    old_dim = config["max_state_dim"]
    if args.max_state_dim < old_dim:
        raise ValueError(f"--max_state_dim ({args.max_state_dim}) must be >= current max_state_dim ({old_dim})")

    state_dict = load_file(src_dir / "model.safetensors")

    weight_key = "model.state_proj.weight"
    weight = state_dict[weight_key]
    hidden_size, current_dim = weight.shape
    if current_dim != old_dim:
        raise ValueError(f"Unexpected {weight_key} shape {weight.shape}, expected last dim {old_dim}")

    padded = torch.zeros((hidden_size, args.max_state_dim), dtype=weight.dtype)
    padded[:, :current_dim] = weight
    state_dict[weight_key] = padded

    config["max_state_dim"] = args.max_state_dim

    save_file(state_dict, output_dir / "model.safetensors")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Copy remaining files (preprocessor/postprocessor normalizer configs, etc.) unchanged.
    for item in src_dir.iterdir():
        if item.name in {"model.safetensors", "config.json"} or item.name.startswith("."):
            continue
        dst = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dst)

    print(f"Wrote expanded checkpoint (max_state_dim {old_dim} -> {args.max_state_dim}) to {output_dir}")
    print(f"Use with: --policy.path={output_dir}")


if __name__ == "__main__":
    main()
