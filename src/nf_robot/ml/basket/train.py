#!/usr/bin/env python

"""Train the basket centering model.

Usage:
    python -m nf_robot.ml.basket.train \\
        --data_root datasets/basket_centering --epochs 30
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from nf_robot.ml.basket.dataset import BasketDataset
from nf_robot.ml.basket.model import (
    BASKET_MODEL_REPOID, LOCAL_MODEL_PATH, BasketNet, decode, return_body_of,
)
from nf_robot.ml.gripper_grid import (
    CELL_SIGMA, DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE, position_losses,
)
from nf_robot.ml.grid_head import masked_mean
from nf_robot.ml.train_common import (
    format_metrics, param_groups, resolve_data_root, upload_model, warmup_cosine,
)
from nf_robot.ml.visual_servoing.uv_methods import gripper_camera_calibration

# Where the mined dataset lives on the hub, for a run that names no local copy.
DEFAULT_DATASET_ID = "naavox/basket_centering"
# Eval metric used by --select_best; lower is better.
SELECTION_METRIC = "lateral_median_cm"
WEIGHTS = {"cell": 1.0, "centroid": 1.0, "distance": 0.5, "rise": 1.0}
RADII_CM = (3, 5, 10)


def basket_loss(outputs, batch, grid, cell_sigma=CELL_SIGMA):
    parts, _, _ = position_losses(
        outputs["logits"], outputs["log_distance"], batch["target_uv"],
        batch["target_range_m"], batch["has_uv"], grid, cell_sigma)
    # Metres, Huber at 5cm: the vertical moves run to half a metre either way.
    parts["rise"], _ = masked_mean(
        F.smooth_l1_loss(outputs["rise"], batch["return_body"][:, 2], reduction="none",
                         beta=0.05), batch["has_uv"])
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def move_metrics(errors, labels, prefix=""):
    """Lateral and vertical error of a move, in cm, against what staying put would score."""
    errors, labels = np.asarray(errors) * 100, np.asarray(labels) * 100
    lateral = np.linalg.norm(errors[:, :2], axis=1)
    out = {
        f"{prefix}lateral_median_cm": float(np.median(lateral)),
        f"{prefix}vertical_median_cm": float(np.median(np.abs(errors[:, 2]))),
        f"{prefix}stay_put_lateral_cm": float(np.median(np.linalg.norm(labels[:, :2], axis=1))),
    }
    for radius in RADII_CM:
        out[f"{prefix}lateral@{radius}cm"] = float((lateral <= radius).mean())
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    """Pixel error of the drop point, and the error of the decoded move."""
    model.eval()
    calibration = gripper_camera_calibration()
    width, height = model.image_size
    scale = torch.tensor([width, height], dtype=torch.float32)
    pixel, moves, labels = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        uv, distance, rise, _ = decode(model(batch["image"], batch["state"]), model.grid)
        pixel.append(((uv - batch["target_uv"]).cpu() * scale).norm(dim=-1))
        moves.append(return_body_of(uv.cpu().numpy(), distance.cpu().numpy(),
                                    rise.float().cpu().numpy(), calibration))
        labels.append(batch["return_body"].cpu().numpy())
    moves, labels = np.concatenate(moves), np.concatenate(labels)
    pixel = torch.cat(pixel)
    return {
        "median_px": pixel.median().item(),
        **move_metrics(moves - labels, labels),
    }


def checkpoint_payload(model, args, metrics, epoch):
    return {
        "state_dict": model.state_dict(),
        "backbone_id": args.backbone,
        "image_size": tuple(args.image_size),
        "fuse_layers": args.fuse_layers,
        "attention_layers": args.attention_layers,
        "freeze": not args.unfreeze_backbone,
        "metrics": metrics,
        "epoch": epoch,
        "cell_sigma": args.cell_sigma,
    }


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    data_root = resolve_data_root(args.data_root, args.dataset_id)
    train_set = BasketDataset(data_root, "train", augment=True)
    eval_set = (BasketDataset(data_root, "eval", augment=False)
                if (data_root / "eval").exists() else None)
    logging.info(f"train {len(train_set)} row(s) | eval {len(eval_set) if eval_set else 'none'}")
    if args.select_best and eval_set is None:
        raise SystemExit(f"--select_best needs an eval split and {data_root}/eval is not built")

    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_set) > args.batch_size, **loader_kwargs)
    eval_loader = (torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        if eval_set else None)

    model = BasketNet(
        backbone_id=args.backbone, image_size=tuple(args.image_size),
        fuse_layers=args.fuse_layers, attention_layers=args.attention_layers,
        freeze=not args.unfreeze_backbone,
    ).to(device)
    groups = param_groups(model, args.lr, args.unfreeze_backbone, args.backbone_lr_scale)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    schedule = warmup_cosine(optimizer, max(1, len(train_loader)) * args.epochs)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best, best_epoch = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        totals, seen = {}, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = model(batch["image"], batch["state"])
            outputs = {k: v.float() for k, v in outputs.items()}
            loss, parts = basket_loss(outputs, batch, model.grid, args.cell_sigma)
            loss.backward()
            optimizer.step()
            schedule.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            seen += 1

        line = f"epoch {epoch + 1}/{args.epochs} " + format_metrics(
            {k: v / max(seen, 1) for k, v in totals.items()})
        metrics = evaluate(model, eval_loader, device) if eval_loader is not None else {}
        logging.info(f"{line} | {format_metrics(metrics)}" if metrics else line)

        if not args.select_best:
            torch.save(checkpoint_payload(model, args, metrics, epoch + 1), args.model_path)
        elif metrics[SELECTION_METRIC] < best:
            best, best_epoch = metrics[SELECTION_METRIC], epoch + 1
            torch.save(checkpoint_payload(model, args, metrics, epoch + 1), args.model_path)
            logging.info(f"saved {args.model_path} ({SELECTION_METRIC} {best:.2f})")

    if args.select_best:
        logging.info(f"done; kept epoch {best_epoch} of {args.epochs} by {SELECTION_METRIC} "
                     f"{best:.2f}, checkpoint at {args.model_path}")
    else:
        logging.info(f"done; {args.epochs} epoch(s), last one kept, checkpoint at {args.model_path}")

    if args.upload:
        upload_model(args.model_path, args.model_id)


def main():
    # force=True because importing transformers installs a root handler.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=None,
                        help="Local mined dataset directory (default: download --dataset_id)")
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--model_path", default=LOCAL_MODEL_PATH)
    parser.add_argument("--select_best", action="store_true",
                        help=f"Keep the epoch with the lowest {SELECTION_METRIC} on eval "
                             f"instead of the last one")
    parser.add_argument("--model_id", default=BASKET_MODEL_REPOID,
                        help="Hub model repo to push the checkpoint to with --upload")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--image_size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fuse_layers", type=int, default=4)
    parser.add_argument("--attention_layers", type=int, default=3)
    parser.add_argument("--cell_sigma", type=float, default=CELL_SIGMA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
