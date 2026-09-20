#!/usr/bin/env python

"""Train the drop point model on a drop pair dataset.

Usage:
    python -m nf_robot.ml.placer.train --data_root datasets/drop_pairs \
        --epochs 30 --batch_size 24

The constant-prediction baseline is printed first and is the number that matters: a room
has a handful of drop points and "always the hamper" is a strong answer, so a model that
does not clearly beat it has learned nothing about the item it was shown.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nf_robot.ml.placer.dataset import EVAL_FRACTION, DropPairDataset
from nf_robot.ml.placer.model import (
    CENTROID_RADIUS, MERGE_MODES, DropPointNet, decode, uv_to_cell, uv_to_metres)
from nf_robot.ml.train_common import param_groups, warmup_cosine
from nf_robot.ml.visual_servoing.model import DEFAULT_BACKBONE, local_centroid
from nf_robot.ml.visual_servoing.train import cell_loss

DEFAULT_MODEL_PATH = "models/drop_point.pth"
SELECTION_METRIC = "median_cm"
# Cells are 15.6cm of floor, and a drop point is a region rather than a point - the
# operator drops anywhere over the hamper - so the target is spread over a couple of them.
CELL_SIGMA = 1.0
CENTROID_WEIGHT = 1.0
RADII_CM = (15, 30, 60)


def drop_loss(outputs, batch, grid, cell_sigma=CELL_SIGMA, centroid_weight=CENTROID_WEIGHT):
    """Cross-entropy over cells against a softened target, plus the windowed centre of mass."""
    logits = outputs["logits"]
    rows, cols = grid
    cell = uv_to_cell(batch["target_uv"], grid)
    cx = cell[:, 0].floor().clamp(0, cols - 1).long()
    cy = cell[:, 1].floor().clamp(0, rows - 1).long()
    index = cy * cols + cx

    parts = {"cell": cell_loss(logits, cell, grid, index, cell_sigma).mean()}
    centroid, _, _ = local_centroid(logits, index, radius=CENTROID_RADIUS)
    reachable = cell.clamp(min=0.5).minimum(cell.new_tensor([cols - 0.5, rows - 0.5]))
    parts["centroid"] = F.smooth_l1_loss(centroid, reachable, reduction="none").mean()
    total = parts["cell"] + centroid_weight * parts["centroid"]
    return total, {k: float(v.detach()) for k, v in parts.items()}


@torch.no_grad()
def evaluate(model, loader, device, radii_cm=RADII_CM, top_k=3, shuffle_items=False):
    """Distance from the predicted drop point to the one the operator used, in centimetres.

    `shuffle_items` pairs each room with another row's item: how much worse that is, is how
    much the model reads the item rather than the room's usual drop point.
    """
    model.eval()
    errors, covered, tasks = [], [], []
    for batch in loader:
        item, overhead = batch["item"].to(device), batch["overhead"].to(device)
        if shuffle_items:
            # Rows of one episode sit together, so shuffle rather than roll by one.
            item = item[torch.randperm(item.shape[0], device=item.device)]
        target = batch["target_uv"].to(device)
        uv, _ = decode(model(item, overhead), model.grid, top_k=top_k)
        distance = (uv - target[:, None, :]).norm(dim=-1)
        errors.append(distance[:, 0].cpu())
        covered.append(distance.min(dim=1).values.cpu())
        tasks += list(batch["task"])

    errors = uv_to_metres(torch.cat(errors)) * 100
    covered = uv_to_metres(torch.cat(covered)) * 100
    metrics = {"median_cm": errors.median().item(), "mean_cm": errors.mean().item()}
    for radius in radii_cm:
        metrics[f"recall@{radius}cm"] = (errors <= radius).float().mean().item()
    metrics[f"top{top_k}@{radii_cm[1]}cm"] = (covered <= radii_cm[1]).float().mean().item()
    return metrics, per_task(errors, tasks)


def per_task(errors, tasks):
    """Median error for each task string, since one room's tasks are far from balanced."""
    out = {}
    for task in sorted(set(tasks)):
        picked = errors[torch.tensor([t == task for t in tasks])]
        out[f"{task} (n{len(picked)})"] = picked.median().item()
    return out


def constant_baseline(train_set, eval_set, radii_cm=RADII_CM):
    """Score of always dropping at the mean training drop point."""
    guess = train_set.labelled_uv().mean(axis=0)
    errors = uv_to_metres(np.linalg.norm(eval_set.labelled_uv() - guess, axis=1)) * 100
    out = {"median_cm": float(np.median(errors)), "mean_cm": float(errors.mean())}
    for radius in radii_cm:
        out[f"recall@{radius}cm"] = float((errors <= radius).mean())
    return out


def _format(metrics):
    return "  ".join(f"{k} {v:.3f}" if abs(v) < 1000 else f"{k} {v:.0f}"
                     for k, v in metrics.items())


def checkpoint_payload(model, args, metrics, epoch):
    return {
        "state_dict": model.state_dict(),
        "backbone_id": args.backbone,
        "overhead_size": list(model.overhead_size),
        "item_size": list(model.item_size),
        "fuse_layers": args.fuse_layers,
        "attention_layers": args.attention_layers,
        "merge": args.merge,
        "freeze": not args.unfreeze_backbone,
        "cell_sigma": args.cell_sigma,
        "metrics": metrics,
        "epoch": epoch,
    }


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    data_root = Path(args.data_root)

    train_set = DropPairDataset(data_root, "train", augment=True,
                                eval_fraction=args.eval_fraction, seed=args.split_seed)
    eval_set = DropPairDataset(data_root, "eval", augment=False,
                               eval_fraction=args.eval_fraction, seed=args.split_seed)
    logging.info(f"constant-prediction baseline: {_format(constant_baseline(train_set, eval_set))}")

    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_set) > args.batch_size, **loader_kwargs)
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = DropPointNet(
        backbone_id=args.backbone, fuse_layers=args.fuse_layers,
        attention_layers=args.attention_layers, merge=args.merge,
        freeze=not args.unfreeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        param_groups(model, args.lr, args.unfreeze_backbone, args.backbone_lr_scale),
        weight_decay=args.weight_decay)
    schedule = warmup_cosine(optimizer, max(1, len(train_loader)) * args.epochs)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, seen = {}, 0
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = model(batch["item"], batch["overhead"])
            outputs = {k: v.float() for k, v in outputs.items()}
            loss, parts = drop_loss(outputs, batch, model.grid, args.cell_sigma,
                                    args.centroid_weight)
            loss.backward()
            optimizer.step()
            schedule.step()
            seen += 1
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v

        line = f"epoch {epoch}/{args.epochs} " + _format({k: v / seen for k, v in totals.items()})
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, by_task = evaluate(model, eval_loader, device)
            logging.info(f"{line} | {_format(metrics)}")
            logging.info("  median_cm by task: " + _format(by_task))
            score = metrics[SELECTION_METRIC]
            if not args.select_best or score < best:
                best = min(best, score)
                torch.save(checkpoint_payload(model, args, metrics, epoch), args.model_path)
                logging.info(f"saved {args.model_path} ({SELECTION_METRIC} {score:.1f})")
        else:
            logging.info(line)
    if args.item_check:
        shuffled, _ = evaluate(model, eval_loader, device, shuffle_items=True)
        logging.info("with each room paired with another row's item: " + _format(shuffled))
    logging.info(f"done; best eval {SELECTION_METRIC} {best:.1f}, checkpoint at {args.model_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="datasets/drop_pairs",
                        help="drop pair dataset directory, holding the all/ pool")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--merge", choices=MERGE_MODES, default="both",
                        help="where the item reaches the room map: FiLM, context tokens, or both")
    parser.add_argument("--eval_fraction", type=float, default=EVAL_FRACTION,
                        help="share of episodes held out")
    parser.add_argument("--split_seed", type=int, default=0, help="which episodes are held out")
    parser.add_argument("--cell_sigma", type=float, default=CELL_SIGMA,
                        help="width in cells of the Gaussian the cell head trains against")
    parser.add_argument("--centroid_weight", type=float, default=CENTROID_WEIGHT)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--fuse_layers", type=int, default=4)
    parser.add_argument("--attention_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--item_check", action="store_true",
                        help="after training, score again with the items shuffled against the "
                             "rooms, to see how much of the answer comes from the item")
    parser.add_argument("--select_best", action="store_true",
                        help="keep the best epoch instead of the last")
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
