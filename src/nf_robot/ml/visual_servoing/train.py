#!/usr/bin/env python

"""Train the visual servoing model, each head masked by its own label mask.

Usage:
    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing \
        --epochs 40 --batch_size 32

    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing_pool_252 \
        --backbone facebook/dinov2-with-registers-base --image_size 448 252 \
        --epochs 40 --batch_size 32
"""

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nf_robot.ml.train_common import param_groups, resolve_data_root, warmup_cosine
from nf_robot.ml.visual_servoing.dataset import VisualServoDataset
from nf_robot.ml.visual_servoing.model import (
    DEFAULT_BACKBONE,
    DEFAULT_IMAGE_SIZE,
    VisualServoNet,
    decode,
    gather_cells,
    load_checkpoint,  # noqa: F401  (re-exported for callers that only import this module)
    local_centroid,
    uv_to_cell,
    window_average,
)

DEFAULT_MODEL_PATH = "models/visual_servo.pth"
# Where the mined dataset lives on the hub, for a run that names no local copy.
DEFAULT_DATASET_ID = "naavox/visual_servoing_dataset"
# Where a trained checkpoint is pushed, with --upload.
DEFAULT_MODEL_ID = "naavox/visual_servo"
# Relative loss weights, so the easy flags don't dominate.
# Eval metric reported each epoch and used by --select_best, on rows whose target is in frame.
SELECTION_METRIC = "onscreen_recall@25px"
DEFAULT_WEIGHTS = {
    "cell": 1.0, "centroid": 1.0, "distance": 0.5,
    "axis": 0.5, "finger": 0.5, "present": 0.2, "holding": 0.2,
    # Only nonzero for a --close_heads model; pressure's small raw magnitude needs the
    # weight.
    "close": 0.2, "pressure": 1.0,
}
# Width in cells (~18px) of the Gaussian the cell head trains against, matched to label
# precision; 0 is one-hot.
CELL_SIGMA = 1.0

# Ten-degree angle bins used to re-weight the axis loss.
AXIS_BINS = 18
# Cap on a bin's weight relative to the average row.
AXIS_WEIGHT_CAP = 10.0
# Largest concentration the axis head may claim, so the axis term can't dominate.
KAPPA_MAX = 50.0


def masked_mean(values, mask):
    """Mean over rows weighted by `mask`; zero when there are none."""
    total = mask.sum()
    return (values * mask).sum() / total.clamp(min=1.0), total


def axis_bin(angle, bins=AXIS_BINS):
    """Which angle bin a pi-periodic axis label falls in, over -pi/2..pi/2."""
    scaled = (angle / math.pi + 0.5) * bins
    return scaled.floor().clamp(0, bins - 1).long()


def axis_bin_weights(angles, bins=AXIS_BINS, cap=AXIS_WEIGHT_CAP):
    """Capped inverse-frequency per-bin weights with mean 1, undoing the axis labels' pile-
    up at zero."""
    angles = np.asarray(angles, dtype=np.float64)
    if not len(angles):
        return np.ones(bins, dtype=np.float32)
    index = np.clip(((angles / math.pi + 0.5) * bins).astype(int), 0, bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    # empty bins keep weight 1: no rows carry it, so the value never applies
    weights = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 1.0)
    weights = np.clip(weights / np.average(weights, weights=counts / counts.sum()), None, cap)
    # renormalize after the clamp so the mean row weight is 1 again
    weights /= np.average(weights, weights=counts / counts.sum())
    return weights.astype(np.float32)


def von_mises_axis_loss(predicted, angle, kappa_max=KAPPA_MAX):
    """Negative log likelihood of a pi-periodic angle under the von Mises the head's (sin
    2t, cos 2t) vector defines."""
    target = torch.stack([torch.sin(2 * angle), torch.cos(2 * angle)], dim=1)
    # Rescale the whole vector to kappa_max, keeping its direction, so length can't be an
    # unbounded reward.
    norm = predicted.norm(dim=1, keepdim=True).clamp(min=1e-6)
    bounded = predicted * (norm.clamp(max=kappa_max) / norm)
    kappa = bounded.norm(dim=1)
    # log I0 via the exponentially scaled Bessel, to stay finite at high concentration.
    log_i0 = torch.special.i0e(kappa).log() + kappa
    return log_i0 - (bounded * target).sum(dim=1)


def soft_cell_target(cell, grid, sigma):
    """A normalized Gaussian over the cell grid centred on the true position."""
    rows, cols = grid
    xs = torch.arange(cols, device=cell.device, dtype=cell.dtype).view(1, 1, cols) + 0.5
    ys = torch.arange(rows, device=cell.device, dtype=cell.dtype).view(1, rows, 1) + 0.5
    squared = ((xs - cell[:, 0].view(-1, 1, 1)) ** 2
               + (ys - cell[:, 1].view(-1, 1, 1)) ** 2)
    target = torch.exp(-0.5 * squared / (sigma * sigma))
    return target.flatten(1) / target.flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12)


def cell_loss(logits, cell, grid, index, sigma):
    """KL divergence of the cell head against a hard or softened target."""
    if sigma <= 0:
        return F.cross_entropy(logits.flatten(1), index, reduction="none")
    target = soft_cell_target(cell, grid, sigma)
    log_probs = F.log_softmax(logits.flatten(1), dim=1)
    entropy = -(target * target.clamp(min=1e-12).log()).sum(dim=1)
    return -(target * log_probs).sum(dim=1) - entropy


def servo_loss(outputs, batch, grid, weights=None, cell_sigma=CELL_SIGMA,
               axis_loss="vonmises", axis_bin_weight=None):
    """Total loss and its parts, each averaged only over rows that carry that label."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    logits = outputs["logits"]
    rows, cols = grid

    cell = uv_to_cell(batch["target_uv"], grid)
    cx = cell[:, 0].floor().clamp(0, cols - 1).long()
    cy = cell[:, 1].floor().clamp(0, rows - 1).long()
    index = cy * cols + cx
    has_uv = batch["has_uv"]

    parts = {}
    # Only the cell head is softened.
    parts["cell"], _ = masked_mean(
        cell_loss(logits, cell, grid, index, cell_sigma), has_uv)

    # Train the windowed centre of mass directly, clamped to the outermost cell centres.
    centroid, window_weights, window = local_centroid(logits, index)
    reachable = cell.clamp(min=0.5).minimum(cell.new_tensor([cols - 0.5, rows - 0.5]))
    parts["centroid"], _ = masked_mean(
        F.smooth_l1_loss(centroid, reachable, reduction="none").mean(dim=1), has_uv)

    # Log metres, since range error is relative.
    predicted_log = gather_cells(outputs["log_distance"].unsqueeze(1), index).squeeze(-1)
    target_log = batch["target_range_m"].clamp(min=1e-3).log()
    parts["distance"], _ = masked_mean(
        F.smooth_l1_loss(predicted_log, target_log, reduction="none"), has_uv)

    angle = batch["grasp_axis_rad"]
    # Average the axis over decode's window with detached weights, so the axis loss can't
    # move probability mass.
    axis = window_average(outputs["axis"], window_weights.detach(), window)
    if axis_loss == "mse":
        axis_target = torch.stack([torch.sin(2 * angle), torch.cos(2 * angle)], dim=1)
        axis_terms = F.mse_loss(axis, axis_target, reduction="none").mean(dim=1)
    else:
        axis_terms = von_mises_axis_loss(axis, angle)
    axis_mask = batch["has_axis"]
    if axis_bin_weight is not None:
        # Folded into the mask so masked_mean divides by the applied weight.
        axis_mask = axis_mask * axis_bin_weight.to(angle.device)[axis_bin(angle)]
    parts["axis"], _ = masked_mean(axis_terms, axis_mask)

    parts["finger"], _ = masked_mean(
        F.smooth_l1_loss(outputs["finger"], batch["finger"], reduction="none"),
        batch["has_finger"])
    parts["present"], _ = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["present_logit"], batch["present"], reduction="none"),
        batch["has_present"])
    parts["holding"], _ = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["holding_logit"], batch["holding"], reduction="none"),
        batch["has_holding"])

    if "close_logit" in outputs:
        parts["close"], _ = masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["close_logit"], batch["close_now"], reduction="none"),
            batch["has_close"])
        # Huber, since the pressure label's tail is measurement noise.
        parts["pressure"], _ = masked_mean(
            F.smooth_l1_loss(outputs["grasp_pressure"], batch["grasp_pressure"],
                             reduction="none", beta=0.05),
            batch["has_pressure"])

    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def wrap_half_pi(radians):
    """Fold an angle difference into -pi/2..pi/2, where a pi-periodic axis lives."""
    return (radians + math.pi / 2) % math.pi - math.pi / 2


@torch.no_grad()
def evaluate(model, loader, device, image_size, radii_px=(10, 25, 50)):
    """Position error in pixels of the input frame, plus each other head's own metric."""
    model.eval()
    width, height = image_size
    scale = torch.tensor([width, height], dtype=torch.float32)

    errors, axis_errors, range_ratio, axis_kappa, axis_labels = [], [], [], [], []
    onscreen = []
    finger_abs, present_ok, holding_ok = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["image"], batch["state"])
        uv, distance, angle, _, concentration = decode(outputs, model.grid, top_k=1)
        uv, distance, angle = uv[:, 0], distance[:, 0], angle[:, 0]
        concentration = concentration[:, 0]

        has_uv = batch["has_uv"] > 0.5
        if has_uv.any():
            delta = (uv - batch["target_uv"])[has_uv].cpu() * scale
            errors.append(delta.norm(dim=-1))
            # Whether the target was in frame at all.
            target = batch["target_uv"][has_uv].cpu()
            onscreen.append(((target - 0.5).abs() <= 0.5).all(dim=-1))
            ratio = distance[has_uv] / batch["target_range_m"][has_uv].clamp(min=1e-3)
            range_ratio.append(ratio.cpu())
        has_axis = batch["has_axis"] > 0.5
        if has_axis.any():
            diff = wrap_half_pi(angle[has_axis] - batch["grasp_axis_rad"][has_axis])
            axis_errors.append(diff.abs().cpu())
            axis_kappa.append(concentration[has_axis].cpu())
            axis_labels.append(batch["grasp_axis_rad"][has_axis].abs().cpu())
        has_finger = batch["has_finger"] > 0.5
        if has_finger.any():
            finger_abs.append((outputs["finger"][has_finger] - batch["finger"][has_finger]).abs().cpu())
        has_present = batch["has_present"] > 0.5
        if has_present.any():
            predicted = (outputs["present_logit"][has_present] > 0).float()
            present_ok.append((predicted == batch["present"][has_present]).float().cpu())
        has_holding = batch["has_holding"] > 0.5
        if has_holding.any():
            predicted = (outputs["holding_logit"][has_holding] > 0).float()
            holding_ok.append((predicted == batch["holding"][has_holding]).float().cpu())

    metrics = {}
    if errors:
        errors = torch.cat(errors)
        metrics["median_px"] = errors.median().item()
        metrics["mean_px"] = errors.mean().item()
        for radius in radii_px:
            metrics[f"recall@{radius}px"] = (errors <= radius).float().mean().item()
        # The same over rows whose target is in frame, the number to select on.
        visible = torch.cat(onscreen)
        if visible.any():
            seen = errors[visible]
            metrics["onscreen_frac"] = visible.float().mean().item()
            metrics["onscreen_median_px"] = seen.median().item()
            for radius in radii_px:
                metrics[f"onscreen_recall@{radius}px"] = (seen <= radius).float().mean().item()
    if range_ratio:
        metrics["range_ratio"] = torch.cat(range_ratio).median().item()
    if axis_errors:
        metrics["axis_deg"] = math.degrees(torch.cat(axis_errors).median().item())
        # What "always upright" scores on the same rows.
        metrics["axis_deg_flat"] = math.degrees(torch.cat(axis_labels).median().item())
        # Median axis concentration; near zero means the head hedges.
        metrics["axis_kappa"] = torch.cat(axis_kappa).median().item()
    if finger_abs:
        metrics["finger_mae"] = torch.cat(finger_abs).mean().item()
    if present_ok:
        metrics["present_acc"] = torch.cat(present_ok).mean().item()
    if holding_ok:
        metrics["holding_acc"] = torch.cat(holding_ok).mean().item()
    return metrics


def constant_baseline(train_set, eval_set, image_size, radii_px=(10, 25, 50)):
    """Score of always predicting the mean training target position."""
    train_uv = train_set.labelled_uv()
    eval_uv = eval_set.labelled_uv()
    if not len(train_uv) or not len(eval_uv):
        return {}
    scale = np.array(image_size, dtype=np.float32)
    errors = np.linalg.norm((eval_uv - train_uv.mean(0)) * scale, axis=1)
    out = {"median_px": float(np.median(errors)), "mean_px": float(errors.mean())}
    for radius in radii_px:
        out[f"recall@{radius}px"] = float((errors <= radius).mean())
    return out


def _format(metrics):
    return "  ".join(
        f"{k} {v:.3f}" if abs(v) < 1000 else f"{k} {v:.0f}" for k, v in metrics.items())


def upload_model(path, model_id, metrics=None):
    """Push a trained checkpoint to the hub, creating the repo if needed."""
    from huggingface_hub import HfApi, create_repo

    path = Path(path)
    create_repo(model_id, repo_type="model", exist_ok=True)
    HfApi().upload_file(
        path_or_fileobj=str(path), path_in_repo=path.name,
        repo_id=model_id, repo_type="model",
        commit_message=f"visual servoing checkpoint ({_format(metrics or {})})",
    )
    logging.info(f"uploaded {path.name} to {model_id}")


def checkpoint_payload(model, args, metrics, epoch):
    """What load_checkpoint needs to rebuild this model, plus how it scored."""
    return {
        "state_dict": model.state_dict(),
        "backbone_id": args.backbone,
        "image_size": tuple(args.image_size),
        "fuse_layers": args.fuse_layers,
        "attention_layers": args.attention_layers,
        # Whether the state dict holds a backbone at all.
        "freeze": not args.unfreeze_backbone,
        "metrics": metrics,
        "epoch": epoch,
        # not needed to rebuild the model, kept so a checkpoint says how it was trained
        "cell_sigma": args.cell_sigma,
        # Record the axis loss, since MSE checkpoints answer zero far more often.
        "axis_loss": args.axis_loss,
        "axis_balance": args.axis_balance,
        # Close/pressure heads; absent in older checkpoints, which load without them.
        "close_heads": args.close_heads,
        # Where the close head reads from; absent in older checkpoints, which use the global
        # one.
        "spatial_close": args.spatial_close,
        # Whether the spatial heads read the pre-attention map; absent in older checkpoints.
        "skip": args.skip,
    }


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    data_root = resolve_data_root(args.data_root, args.dataset_id)
    logging.info(f"dataset at {data_root}")

    train_set = VisualServoDataset(data_root, "train", augment=True)
    eval_set = (VisualServoDataset(data_root, "eval", augment=False)
                if (data_root / "eval").exists() else None)
    logging.info(f"train {len(train_set)} row(s) | "
                 f"eval {len(eval_set) if eval_set else 'none built'}")
    if args.select_best and eval_set is None:
        raise SystemExit(f"--select_best needs an eval split and {data_root}/eval is not built")

    axis_bin_weight = None
    if args.axis_balance:
        labels = train_set.labelled_axis()
        weights_np = axis_bin_weights(labels)
        axis_bin_weight = torch.from_numpy(weights_np)
        near_zero = float(np.mean(np.abs(np.degrees(labels)) < 5.0)) if len(labels) else 0.0
        logging.info(f"axis balance over {AXIS_BINS} bins from {len(labels)} labels "
                     f"({near_zero:.0%} within 5 deg of zero): weights "
                     f"{weights_np.min():.2f}-{weights_np.max():.2f}")

    image_size = tuple(args.image_size)
    if eval_set:
        baseline = constant_baseline(train_set, eval_set, image_size)
        if baseline:
            logging.info(f"constant-prediction baseline: {_format(baseline)}")

    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_set) > args.batch_size, **loader_kwargs)
    eval_loader = (torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        if eval_set else None)

    if args.close_heads and not train_set.has_close_labels():
        raise SystemExit(
            f"The close heads need close_now labels and no row in {data_root}/train has "
            f"one. Re-mine the teleop half (the columns are written by mine_teleop), or "
            f"pass --no_close_heads to train the finger-rate head alone.")
    model = VisualServoNet(
        backbone_id=args.backbone, image_size=image_size, fuse_layers=args.fuse_layers,
        close_heads=args.close_heads, spatial_close=args.spatial_close, skip=args.skip,
        attention_layers=args.attention_layers, freeze=not args.unfreeze_backbone,
    ).to(device)
    groups = param_groups(model, args.lr, args.unfreeze_backbone, args.backbone_lr_scale)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    schedule = warmup_cosine(optimizer, max(1, len(train_loader)) * args.epochs)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best, best_epoch = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        totals, seen = {}, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = model(batch["image"], batch["state"])
            outputs = {k: v.float() for k, v in outputs.items()}
            loss, parts = servo_loss(outputs, batch, model.grid, cell_sigma=args.cell_sigma,
                                     axis_loss=args.axis_loss,
                                     axis_bin_weight=axis_bin_weight)
            loss.backward()
            optimizer.step()
            schedule.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            seen += 1

        line = f"epoch {epoch + 1}/{args.epochs} " + _format({k: v / max(seen, 1) for k, v in totals.items()})

        # Eval is reported every epoch but only picks the checkpoint with --select_best,
        # since it is a narrow proxy.
        metrics = {}
        due = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if due and eval_loader is not None:
            metrics = evaluate(model, eval_loader, device, image_size)
            logging.info(f"{line} | {_format(metrics)}")
        else:
            logging.info(line)

        if not args.select_best:
            torch.save(checkpoint_payload(model, args, metrics, epoch + 1), args.model_path)
        elif metrics:
            score = metrics.get(SELECTION_METRIC, -1.0)
            if score > best:
                best, best_epoch = score, epoch + 1
                torch.save(checkpoint_payload(model, args, metrics, epoch + 1),
                           args.model_path)
                logging.info(f"saved {args.model_path} ({SELECTION_METRIC} {score:.3f})")

    if args.select_best:
        # Loud, because a selected checkpoint is older than the run.
        logging.info(f"done; kept epoch {best_epoch} of {args.epochs} by "
                     f"{SELECTION_METRIC} {best:.3f}, checkpoint at {args.model_path}")
    else:
        logging.info(f"done; {args.epochs} epoch(s), last one kept, "
                     f"checkpoint at {args.model_path}")

    if args.upload:
        if not Path(args.model_path).exists():
            logging.error(f"nothing to upload: no checkpoint at {args.model_path}")
        else:
            upload_model(args.model_path, args.model_id)


def main():
    # force=True because importing lerobot/transformers installs a root handler.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=None,
                        help="Local mined dataset directory (default: download --dataset_id)")
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                        help="Mined dataset on the hub, used when --data_root is absent")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    # Close heads are on by default; --close_heads is still accepted.
    parser.add_argument(
        "--close_heads", dest="close_heads", action="store_true", default=True,
        help="Train the close-onset and grasp-pressure heads (the default). The "
             "finger-rate head still trains alongside them.")
    parser.add_argument(
        "--no_close_heads", dest="close_heads", action="store_false",
        help="Train the finger-rate head alone, the way checkpoints before the close "
             "heads existed were built. Needed for a pool with no close_now labels.")
    # None so an explicit request can be told from the default.
    parser.add_argument(
        "--spatial_close", dest="spatial_close", action="store_true", default=None,
        help="Read the close head off the patch grid rather than the pooled [CLS] vector "
             "(the default whenever the close heads are built). Whether to close is a "
             "spatial test - something between the fingers, square on, near enough - and "
             "a vector averaged over the whole image can say the frame holds a graspable "
             "thing but not that this one is in the jaws.")
    parser.add_argument(
        "--global_close", dest="spatial_close", action="store_false",
        help="Read the close head off the [CLS] vector instead, the way it was built "
             "before the spatial head.")
    parser.add_argument(
        "--no_skip", dest="skip", action="store_false",
        help="Leave out the skip connection from the pre-attention map to the spatial "
             "heads, for the A/B against the model that averaged several objects' positions.")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Score the eval split every N epochs (and always on the last)")
    parser.add_argument("--select_best", action="store_true",
                        help=f"Keep the epoch that scores best on {SELECTION_METRIC} "
                             f"instead of the last one. Off by default: the eval split is "
                             f"one held-out room measured on a position proxy, and the "
                             f"heads it cannot see are still learning after it stops "
                             f"improving.")
    parser.add_argument("--axis_loss", default="vonmises", choices=["vonmises", "mse"],
                        help="Objective for the grasp axis head. vonmises trains the "
                             "output's length as a confidence and charges for hedging; "
                             "mse is the previous behaviour, kept for the A/B")
    parser.add_argument("--no_axis_balance", dest="axis_balance", action="store_false",
                        help="Weight every axis row alike instead of by angle bin. The "
                             "labels lean hard on zero, so this is the old behaviour")
    parser.set_defaults(axis_balance=True)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="Hub model repo to push the checkpoint to with --upload")
    parser.add_argument("--upload", action="store_true",
                        help="Push the best checkpoint to --model_id when training ends")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--image_size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fuse_layers", type=int, default=4)
    parser.add_argument("--attention_layers", type=int, default=3)
    parser.add_argument("--cell_sigma", type=float, default=CELL_SIGMA,
                        help="Width in cells of the Gaussian the cell head is trained "
                             "against; 0 for the old one-hot target")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.spatial_close and not args.close_heads:
        parser.error("--spatial_close moves the close head that --no_close_heads just "
                     "switched off; pass one or the other.")
    # The unset default follows the close heads.
    if args.spatial_close is None:
        args.spatial_close = args.close_heads
    train(args)


if __name__ == "__main__":
    main()
