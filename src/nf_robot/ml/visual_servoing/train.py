#!/usr/bin/env python

"""Train the visual servoing model on the mined/synthesised dataset.

Every head is masked by its own label mask, so a row that only knows where the object
is trains only the position heads, and a row that only knows we are holding something
trains only that. This is what lets teleop mining and the synthetic compositor write
into one dataset without either of them inventing labels it cannot know - see
readme.md, and mine_teleop.py for the producer that currently exists.

Validation is real teleop, held out by whole episode: consecutive frames of one
approach are near-duplicates, so splitting inside an episode scores the model on frames
it effectively trained on. The constant-prediction baseline is printed alongside,
because for a centering task "always predict the middle" is an embarrassingly strong
answer and a model that fails to beat it has learned nothing about the image.

Usage:
    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing \
        --epochs 40 --batch_size 32

    On the ungated backbone, against a dataset rebuilt at 448x252 (see readme.md,
    "Training on the ungated backbone" - the stored frame size has to change with the
    backbone's patch size, and nothing checks that it did):

    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing_252 \
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

from nf_robot.ml.visual_servoing.dataset import VisualServoDataset
from nf_robot.ml.visual_servoing.model import (
    DEFAULT_BACKBONE,
    DEFAULT_IMAGE_SIZE,
    VisualServoNet,
    decode,
    gather_cells,
    load_checkpoint,  # noqa: F401  (re-exported for callers that only import this module)
    uv_to_cell,
)

DEFAULT_MODEL_PATH = "models/visual_servo.pth"
# Where the mined dataset lives on the hub, for a run that names no local copy.
DEFAULT_DATASET_ID = "naavox/visual_servoing_dataset"
# Where a trained checkpoint is pushed, with --upload.
DEFAULT_MODEL_ID = "naavox/visual_servo"
# Relative weights. Position is the point of the model; the flags are easy and would
# otherwise dominate a sum of raw losses simply by being confidently right.
DEFAULT_WEIGHTS = {
    "cell": 1.0, "offset": 1.0, "distance": 0.5,
    "axis": 0.5, "finger": 0.5, "present": 0.2, "holding": 0.2,
}
# Width, in cells, of the Gaussian the cell head is trained against. Cells are 12px of the
# 448x256 input (the 1.5x canvas over 56x32 cells), so 1.5 cells is about 18px.
#
# Sized to the labels rather than to the grid. A mined label is a room point projected
# back through the approach, and that projection ignores the gripper's swing and hangs its
# anchor off a rangefinder reading - it is good to a few cells, not to one. Training a
# 1792-way softmax on a one-hot target against a label that imprecise asks the network to
# reproduce the error along with the position, which it can only do by memorising: the
# training cross-entropy falls and nothing transfers. Spreading the target over the cells
# the label plausibly covers asks for what is actually known.
#
# Set to 0 for the old one-hot target, which is the A/B worth running.
CELL_SIGMA = 1.5

# Angle bins used to re-weight the axis loss, over the -pi/2..pi/2 a pi-periodic axis
# lives in. Ten degrees per bin: fine enough to separate "upright" from "a little off",
# coarse enough that a bin still holds a usable number of rows.
AXIS_BINS = 18
# Most any one bin's rows can be worth relative to the average row. Inverse frequency
# without a cap hands a nearly empty bin unbounded pull.
AXIS_WEIGHT_CAP = 10.0
# Largest concentration the axis head may claim. The von Mises likelihood is unbounded in
# kappa for a perfectly predicted angle, and an unbounded kappa is how the axis term comes
# to dominate every other head late in a run.
KAPPA_MAX = 50.0


def masked_mean(values, mask):
    """Mean over the rows a label actually exists for; zero when there are none.

    `mask` doubles as a per-row weight: it is 0/1 for a plain label mask, and the axis
    head passes a re-weighted version of it so that rare angles count for more. The
    denominator is the weight sum either way, which keeps the loss a mean rather than a
    sum that grows with how many rare rows a batch happened to draw.
    """
    total = mask.sum()
    return (values * mask).sum() / total.clamp(min=1.0), total


def axis_bin(angle, bins=AXIS_BINS):
    """Which angle bin a pi-periodic axis label falls in, over -pi/2..pi/2."""
    scaled = (angle / math.pi + 0.5) * bins
    return scaled.floor().clamp(0, bins - 1).long()


def axis_bin_weights(angles, bins=AXIS_BINS, cap=AXIS_WEIGHT_CAP):
    """Per-bin weights that undo the label distribution's lean toward zero.

    69% of the axis labels in a mixed split sit within 5 degrees of zero, and the mined
    half is 85% of the way there on its own, because a teleoperator sets the wrist early
    and "how much further it turns" is zero from then on. Weighting every row equally
    therefore spends almost all of the head's gradient on the one answer it already knows,
    and the rare orientations - the ones the wrist actually has to move for - arrive as
    noise around it.

    Inverse frequency over bins, normalized so the mean weight of the training rows is 1,
    which keeps this from silently rescaling the axis term against the other heads. Capped
    because inverse frequency is unstable in the tail: a bin holding five rows would
    otherwise be handed thousands of times the pull of the bulk, and the head would chase
    those five.
    """
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
    """Negative log likelihood of a pi-periodic angle under a von Mises the head predicts.

    The head emits an unnormalized (sin 2t, cos 2t). Read as a von Mises, its direction is
    the answer and its length is the concentration - how sure the head is - so both halves
    of the output mean something and both are trained.

    That is the whole point of the change. Under the mean squared error this replaces, the
    optimum is the conditional mean of the target vector, and `decode` reads the direction
    with atan2, which does not care how long the vector is. Shrinking toward the mean was
    therefore free, and with labels piled at zero the mean *is* zero: a head that answered
    "0 degrees, always" was sitting at the bottom of its loss. Here, shrinking costs
    likelihood - log I0 rewards length when the direction is right and punishes it when it
    is wrong - so hedging has a price and confidence has a meaning.

    Written as log I0(k) - v.u because the dot product of the raw output with the unit
    target already equals k cos(2t_pred - 2t_true); no atan2 in the loss and no gradient
    through one.
    """
    target = torch.stack([torch.sin(2 * angle), torch.cos(2 * angle)], dim=1)
    # The whole vector is bounded, not just the kappa read off it. Clamping only the
    # log-partition term leaves the dot product below free to grow without limit, which
    # makes an arbitrarily long vector an arbitrarily large reward - the exact runaway
    # kappa_max exists to stop. Rescaling instead keeps the direction and its gradient
    # while the magnitude saturates.
    norm = predicted.norm(dim=1, keepdim=True).clamp(min=1e-6)
    bounded = predicted * (norm.clamp(max=kappa_max) / norm)
    kappa = bounded.norm(dim=1)
    # log I0 via the exponentially scaled Bessel, which is what keeps this finite at the
    # concentrations a confident head reaches
    log_i0 = torch.special.i0e(kappa).log() + kappa
    return log_i0 - (bounded * target).sum(dim=1)


def soft_cell_target(cell, grid, sigma):
    """A Gaussian over the cell grid centred on the true position, as a distribution.

    Centres are at i + 0.5 in the continuous cell coordinates uv_to_cell produces, so the
    peak sits where the label actually falls rather than snapping to the cell it lands in.

    Normalized over the grid after the fact, which matters at the edges: a target near a
    corner has most of its Gaussian outside the canvas, and renormalizing puts that mass
    back on the cells that exist instead of quietly training against a target that sums to
    less than one.
    """
    rows, cols = grid
    xs = torch.arange(cols, device=cell.device, dtype=cell.dtype).view(1, 1, cols) + 0.5
    ys = torch.arange(rows, device=cell.device, dtype=cell.dtype).view(1, rows, 1) + 0.5
    squared = ((xs - cell[:, 0].view(-1, 1, 1)) ** 2
               + (ys - cell[:, 1].view(-1, 1, 1)) ** 2)
    target = torch.exp(-0.5 * squared / (sigma * sigma))
    return target.flatten(1) / target.flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12)


def cell_loss(logits, cell, grid, index, sigma):
    """Cross-entropy of the cell head against a hard or a softened target.

    Reported as a KL divergence rather than a raw cross-entropy: against a soft target the
    cross-entropy bottoms out at the target's own entropy, so the raw number would neither
    reach zero nor compare with a run at another sigma. The gradient is identical - the
    entropy subtracted is a constant of the labels.
    """
    if sigma <= 0:
        return F.cross_entropy(logits.flatten(1), index, reduction="none")
    target = soft_cell_target(cell, grid, sigma)
    log_probs = F.log_softmax(logits.flatten(1), dim=1)
    entropy = -(target * target.clamp(min=1e-12).log()).sum(dim=1)
    return -(target * log_probs).sum(dim=1) - entropy


def servo_loss(outputs, batch, grid, weights=None, cell_sigma=CELL_SIGMA,
               axis_loss="vonmises", axis_bin_weight=None):
    """Total loss and its parts, each averaged only over rows that carry that label.

    axis_loss picks between the von Mises likelihood and the mean squared error it
    replaced, which is the A/B worth running before believing any of the argument above
    it. axis_bin_weight is a per-bin weight vector from axis_bin_weights, or None to
    weight every axis row alike.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    logits = outputs["logits"]
    rows, cols = grid

    cell = uv_to_cell(batch["target_uv"], grid)
    cx = cell[:, 0].floor().clamp(0, cols - 1).long()
    cy = cell[:, 1].floor().clamp(0, rows - 1).long()
    index = cy * cols + cx
    has_uv = batch["has_uv"]

    parts = {}
    # Only the cell head is softened. The offset, distance and axis heads are all read at
    # the one true cell, where a spread target would mean nothing.
    parts["cell"], _ = masked_mean(
        cell_loss(logits, cell, grid, index, cell_sigma), has_uv)

    frac = (cell - torch.stack([cx, cy], dim=1).float()).clamp(0.0, 1.0)
    offset = gather_cells(outputs["offsets"], index).sigmoid()
    parts["offset"], _ = masked_mean(
        F.l1_loss(offset, frac, reduction="none").mean(dim=1), has_uv)

    # Log metres: the useful error in a range is relative, and the head has to cover
    # everything from a gripper across the room to one about to touch the object.
    predicted_log = gather_cells(outputs["log_distance"].unsqueeze(1), index).squeeze(-1)
    target_log = batch["target_range_m"].clamp(min=1e-3).log()
    parts["distance"], _ = masked_mean(
        F.smooth_l1_loss(predicted_log, target_log, reduction="none"), has_uv)

    angle = batch["grasp_axis_rad"]
    axis = gather_cells(outputs["axis"], index)
    if axis_loss == "mse":
        axis_target = torch.stack([torch.sin(2 * angle), torch.cos(2 * angle)], dim=1)
        axis_terms = F.mse_loss(axis, axis_target, reduction="none").mean(dim=1)
    else:
        axis_terms = von_mises_axis_loss(axis, angle)
    axis_mask = batch["has_axis"]
    if axis_bin_weight is not None:
        # folded into the mask rather than the loss, so masked_mean divides by the weight
        # that was actually applied and an unlabelled row still contributes nothing
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
            # Whether the answer was in the picture at all. A target outside the frame
            # cannot be found by looking, so it sets a floor on any average that includes
            # it, and a model can improve for a long time without moving one.
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
        # The same again over the rows the image can actually answer. This is the number
        # that moves when the model learns something, and the one to select a checkpoint
        # on; the headline figures above are diluted by however many blind rows the split
        # happens to carry.
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
        # What "always upright" scores on the same rows. The labels pile up at zero, so
        # that is a strong answer and a head can look good while knowing nothing about the
        # image; printing the two together is what makes the difference legible.
        metrics["axis_deg_flat"] = math.degrees(torch.cat(axis_labels).median().item())
        # Median concentration: how sure the head is, in the units the von Mises loss
        # trains. Near zero is a head that has learned to hedge, which is the failure this
        # objective exists to make visible rather than silent.
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


def resolve_data_root(args) -> Path:
    """The mined dataset on disk, downloading it from the hub if no local copy was named.

    Same shape as ortho_target.resolve_data_root: a local directory wins, and naming a
    hub dataset is what asks for a download, so a mistyped path fails loudly instead of
    quietly fetching something else.
    """
    if args.data_root:
        return Path(args.data_root)
    from huggingface_hub import snapshot_download

    logging.info(f"Downloading {args.dataset_id}")
    return Path(snapshot_download(repo_id=args.dataset_id, repo_type="dataset"))


def upload_model(path, model_id, metrics=None):
    """Push a trained checkpoint to the hub, creating the repo if it does not exist.

    Uploaded once at the end rather than on every improvement: the checkpoint carries
    the frozen backbone's weights too, so it is a few hundred megabytes, and pushing
    that each time the score ticks up would cost more than the training.
    """
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
        # Whether the state dict above holds a backbone at all: a frozen one is left to
        # dino_trunk's shared instance and never written.
        "freeze": not args.unfreeze_backbone,
        "metrics": metrics,
        "epoch": epoch,
        # not needed to rebuild the model, kept so a checkpoint says how it was trained
        "cell_sigma": args.cell_sigma,
        # A checkpoint trained under the mean squared error decodes the same way and
        # answers zero far more often; that is worth being able to tell from the file
        # rather than from whichever run log is still around.
        "axis_loss": args.axis_loss,
        "axis_balance": args.axis_balance,
    }


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    data_root = resolve_data_root(args)
    logging.info(f"dataset at {data_root}")

    # Only the train split. Scoring a held-out room is what evaluate.py is for, and it is
    # a separate command on purpose: it takes a finished checkpoint and says how it did,
    # rather than getting a vote in what the checkpoint is.
    train_set = VisualServoDataset(data_root, "train", augment=True)
    logging.info(f"train {len(train_set)} row(s)")

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
    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_set) > args.batch_size, **loader_kwargs)

    model = VisualServoNet(
        backbone_id=args.backbone, image_size=image_size, fuse_layers=args.fuse_layers,
        attention_layers=args.attention_layers, freeze=not args.unfreeze_backbone,
    ).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if args.unfreeze_backbone:
        groups.append({"params": list(model.trunk.parameters()),
                       "lr": args.lr * args.backbone_lr_scale})
        logging.info(f"backbone unfrozen at {args.backbone_lr_scale}x the head learning rate")
    else:
        logging.info(f"backbone frozen; training {sum(p.numel() for p in head_params) / 1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = max(1, len(train_loader)) * args.epochs
    warmup = max(1, int(0.05 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    ))
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
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

        logging.info(line)
        # Written every epoch, always the newest weights. Nothing here judges a checkpoint
        # and nothing here discards one.
        torch.save(checkpoint_payload(model, args, {}, epoch + 1), args.model_path)

    logging.info(f"done; {args.epochs} epoch(s), checkpoint at {args.model_path}")

    if args.upload:
        if not Path(args.model_path).exists():
            logging.error(f"nothing to upload: no checkpoint at {args.model_path}")
        else:
            upload_model(args.model_path, args.model_id)


def main():
    # force=True: importing lerobot/transformers installs a root handler, which makes a
    # later basicConfig a silent no-op and drops every info line this tool logs.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=None,
                        help="Local mined dataset directory (default: download --dataset_id)")
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                        help="Mined dataset on the hub, used when --data_root is absent")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
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
    train(parser.parse_args())


if __name__ == "__main__":
    main()
