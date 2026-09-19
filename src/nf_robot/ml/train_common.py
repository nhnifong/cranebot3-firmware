"""Training plumbing shared by ortho_target and visual_servoing."""

import logging
import math
from pathlib import Path

import torch


def resolve_data_root(data_root, dataset_id) -> Path:
    """Local dataset directory if named, else a fresh sync of the hub copy."""
    if data_root:
        return Path(data_root)
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))
    logging.info(f"Using {dataset_id} from the hub at {root}")
    return root


def param_groups(model, lr, unfreeze_backbone, backbone_lr_scale):
    """AdamW groups: the head at lr, and the trunk at a fraction of it if it trains at all."""
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": lr}]
    if unfreeze_backbone:
        groups.append({"params": list(model.trunk.parameters()), "lr": lr * backbone_lr_scale})
        logging.info(f"backbone unfrozen at {backbone_lr_scale}x the head learning rate")
    else:
        logging.info(f"backbone frozen; training {sum(p.numel() for p in head_params) / 1e6:.1f}M head parameters")
    return groups


def warmup_cosine(optimizer, steps, warmup_fraction=0.05):
    """Linear warmup over the first warmup_fraction of steps, then cosine decay to zero."""
    warmup = max(1, int(warmup_fraction * steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    ))
