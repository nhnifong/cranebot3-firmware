#!/usr/bin/env python

"""Training and scoring the ortho target model."""

import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from nf_robot.ml.ortho_target.dataset import MAX_TARGETS, OrthoTargetDataset, is_complete
from nf_robot.ml.ortho_target.model import (
    CELL_SIGMA,
    DEFAULT_MODEL_PATH,
    ORTHO_EXTENT_M,
    TARGET_THRESHOLD,
    TARGETING_MODEL_FILENAME,
    TARGETING_MODEL_REPOID,
    OrthoTargetNet,
    load_checkpoint,
    objectness_loss,
    predict,
)
from nf_robot.ml.train_common import param_groups, resolve_data_root, warmup_cosine

# Peaks the scoring counts over, well above any plausible number of floor targets.
COUNT_K = 64
# Detection match radius, and the thresholds swept for the best operating point.
MATCH_RADIUS_CM = 20.0
THRESHOLD_SWEEP = tuple(round(0.05 * i, 2) for i in range(1, 20)) + (0.975, 0.99, 0.995, 0.999)

# Metric that picks the saved checkpoint: detection F1 on complete frames at the best
# threshold.
SELECTION_METRIC = "f1@20cm"


def balanced_pos_weight(dataset, grid, cell_sigma=CELL_SIGMA):
    """Supervised negative cells per unit of positive mass over a dataset, which balances
    the loss."""
    bump = 2.0 * math.pi * cell_sigma ** 2
    positives = sum(min(len(s["points"]), MAX_TARGETS) for s in dataset.samples) * bump
    negatives = sum(grid * grid - min(len(s["points"]), MAX_TARGETS) * bump
                    for s in dataset.samples if is_complete(s))
    if positives <= 0 or negatives <= 0:
        raise ValueError(
            f"cannot balance the objectness loss: {positives / max(bump, 1e-9):.0f} labelled "
            f"target(s) against {negatives:.0f} supervised background cell(s). Without "
            f"complete frames (is_complete) nothing in the dataset says where objects are "
            f"not, and the head has no negatives to learn from.")
    return negatives / positives


def pixels_to_cm(pixels, image_size):
    return pixels * (ORTHO_EXTENT_M * 100.0) / image_size


@torch.no_grad()
def evaluate_model(model, loader, device, top_k=5, tta=False, radii_cm=(10, 20, 50),
                   match_radius_cm=MATCH_RADIUS_CM, thresholds=THRESHOLD_SWEEP):
    """Score a checkpoint: detection metrics on complete frames at the best swept threshold,
    distance metrics over every frame."""
    model.eval()
    match_px = match_radius_cm * model.image_size / (ORTHO_EXTENT_M * 100.0)
    nearest, covered, all_scores, total_bce, count = [], [], [], 0.0, 0
    ranked, labelled = [], 0  # (score, hit) per candidate of a complete frame; its labels

    for images, points, mask, complete in loader:
        images = images.to(device)
        points, mask, complete = points.to(device), mask.to(device), complete.to(device)
        logits, offsets = model(images)
        _, bce, _ = objectness_loss(logits, offsets, points, mask, complete,
                                    model.image_size, model.grid)
        total_bce += bce.item() * images.shape[0]
        count += images.shape[0]

        # One decode wide enough for the count, sliced back to top_k for the ranking.
        uv, scores = predict(model, images, tta=tta, top_k=max(top_k, COUNT_K))
        all_scores.append(scores.cpu())
        # (B, labels, k): every label against every decoded candidate.
        distance = (points[:, :, None, :] - uv[:, None, :top_k, :]).norm(dim=-1)
        real = mask > 0
        nearest.append(distance[:, :, 0][real].cpu())
        covered.append(distance.min(dim=2).values[real].cpu())

        for i in torch.nonzero(complete > 0).flatten().tolist():
            labels = points[i][mask[i] > 0].cpu().numpy()
            labelled += len(labels)
            ranked.extend(match_frame(labels, uv[i].cpu().numpy(), scores[i].cpu().numpy(), match_px))

    nearest = pixels_to_cm(torch.cat(nearest), model.image_size)
    covered = pixels_to_cm(torch.cat(covered), model.image_size)
    metrics = {
        "bce": total_bce / max(count, 1),
        "median_cm": nearest.median().item(),
        "mean_cm": nearest.mean().item(),
    }
    for radius in radii_cm:
        metrics[f"recall@{radius}cm"] = (nearest <= radius).float().mean().item()
    metrics[f"top{top_k}@20cm"] = (covered <= 20).float().mean().item()

    best = best_f1(ranked, labelled, thresholds)
    metrics.update({
        f"f1@{match_radius_cm:.0f}cm": best["f1"],
        f"precision@{match_radius_cm:.0f}cm": best["precision"],
        f"found@{match_radius_cm:.0f}cm": best["found"],
        "threshold": best["threshold"],
        "scored_targets": best["frames"],
    })
    scores = torch.cat(all_scores)
    metrics["targets_per_frame"] = (scores >= best["threshold"]).sum(dim=1).float().mean().item()
    return metrics


def match_frame(labels, uv, scores, radius_px):
    """Greedily match a frame's ranked candidates to its labels, as (score, hit) per candidate."""
    out, taken = [], set()
    for (u, v), score in zip(uv, scores):
        hit = False
        if len(labels):
            d = np.linalg.norm(labels - np.array([u, v]), axis=1)
            for j in np.argsort(d):
                if d[j] > radius_px:
                    break
                if j not in taken:
                    taken.add(int(j))
                    hit = True
                    break
        out.append((float(score), hit))
    return out


def best_f1(ranked, labelled, thresholds=THRESHOLD_SWEEP):
    """The threshold with the best F1 on the complete frames, and what it scores."""
    if not ranked or not labelled:
        return {"f1": 0.0, "precision": 0.0, "found": 0.0,
                "threshold": TARGET_THRESHOLD, "frames": 0}
    scores = np.array([s for s, _ in ranked])
    hits = np.array([h for _, h in ranked])
    best = {"f1": -1.0}
    for t in thresholds:
        above = scores >= t
        tp = int((above & hits).sum())
        fp = int((above & ~hits).sum())
        precision = tp / max(tp + fp, 1)
        found = tp / labelled
        f1 = 2 * precision * found / max(precision + found, 1e-9)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": precision, "found": found, "threshold": float(t)}
    best["frames"] = labelled
    return best


def constant_baseline(train_set, eval_set, image_size, radii_cm=(10, 20, 50)):
    """Score of always predicting the mean training label."""
    train_uv = train_set.scaled_labels()
    eval_uv = eval_set.scaled_labels()
    errors = pixels_to_cm(np.linalg.norm(eval_uv - train_uv.mean(0), axis=1), image_size)
    out = {"median_cm": float(np.median(errors)), "mean_cm": float(errors.mean())}
    for radius in radii_cm:
        out[f"recall@{radius}cm"] = float((errors <= radius).mean())
    return out


def resolve_model_path(model_path) -> str:
    """The checkpoint to evaluate, downloading the published one if the default path is missing."""
    if Path(model_path).exists():
        return model_path
    if model_path != DEFAULT_MODEL_PATH:
        raise FileNotFoundError(f"No checkpoint at {model_path}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=TARGETING_MODEL_REPOID, filename=TARGETING_MODEL_FILENAME)
    logging.info(f"No {model_path}; using {TARGETING_MODEL_REPOID}/{TARGETING_MODEL_FILENAME} at {path}")
    return path


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    data_root = resolve_data_root(args.data_root, args.dataset_id)
    train_set = OrthoTargetDataset(data_root, "train", args.image_size, augment=True,
                                   seed=args.seed, translate_px=args.translate_px)
    eval_set = OrthoTargetDataset(data_root, "eval", args.image_size, augment=False)
    logging.info(f"train {len(train_set)} sample(s) | eval {len(eval_set)} sample(s) from {data_root}")

    complete_eval = sum(1 for sample in eval_set.samples if is_complete(sample))
    if args.select_metric.startswith("f1") and not complete_eval:
        raise ValueError(
            f"{data_root}/eval holds no complete frames, so {args.select_metric} cannot be "
            f"computed: nothing there can tell a false detection from an object nobody "
            f"labelled. Merge hand labels into the pool and deal the splits from it - "
            f"merge_labels then split - or pick a --select_metric from the distance family.")
    pos_weight = args.pos_weight or balanced_pos_weight(train_set, args.grid, args.cell_sigma)
    logging.info(f"objectness pos_weight {pos_weight:.0f}; complete frames: "
                 f"{sum(1 for s in train_set.samples if is_complete(s))} of {len(train_set)} train, "
                 f"{complete_eval} of {len(eval_set)} eval")

    baseline = constant_baseline(train_set, eval_set, args.image_size)
    logging.info(f"constant-prediction baseline: {_format_metrics(baseline)}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        drop_last=len(train_set) > args.batch_size, pin_memory=device.type == "cuda",
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = OrthoTargetNet(
        backbone_id=args.backbone, image_size=args.image_size, grid=args.grid,
        fuse_layers=args.fuse_layers, freeze=not args.unfreeze_backbone,
        attention_layers=args.attention_layers, attention_skip=not args.no_attention_skip,
    ).to(device)
    groups = param_groups(model, args.lr, args.unfreeze_backbone, args.backbone_lr_scale)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    schedule = warmup_cosine(optimizer, max(1, len(train_loader)) * args.epochs)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(3)
        for images, points, mask, complete in train_loader:
            images = images.to(device, non_blocking=True)
            points, mask = points.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            complete = complete.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits, offsets = model(images)
                loss, bce, l1 = objectness_loss(
                    logits.float(), offsets.float(), points, mask, complete,
                    args.image_size, args.grid, args.offset_weight,
                    cell_sigma=args.cell_sigma, pos_weight=pos_weight,
                )
            loss.backward()
            optimizer.step()
            schedule.step()
            totals += [loss.item(), bce.item(), l1.item()]

        totals /= max(1, len(train_loader))
        line = f"epoch {epoch + 1}/{args.epochs} loss {totals[0]:.4f} (bce {totals[1]:.4f} off {totals[2]:.4f})"

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            metrics = evaluate_model(model, eval_loader, device, top_k=args.top_k)
            logging.info(f"{line} | {_format_metrics(metrics)}")
            score = metrics[args.select_metric]
            if score > best:
                best = score
                torch.save({
                    "state_dict": model.state_dict(),
                    "backbone_id": args.backbone,
                    "image_size": args.image_size,
                    "grid": args.grid,
                    "fuse_layers": args.fuse_layers,
                    "attention_layers": args.attention_layers,
                    "attention_skip": model.attention_skip,
                    # What the logits mean, so an old softmax checkpoint can't load
                    # silently.
                    "head": "objectness",
                    # The best operating threshold and the pos_weight it depends on travel
                    # with the weights.
                    "threshold": metrics["threshold"],
                    "pos_weight": pos_weight,
                    # Whether the state dict holds a backbone at all.
                    "freeze": not args.unfreeze_backbone,
                    "metrics": metrics,
                    "epoch": epoch + 1,
                }, args.model_path)
                logging.info(f"saved {args.model_path} ({args.select_metric} {score:.3f})")
        else:
            logging.info(line)

    logging.info(f"done; best eval {args.select_metric} {best:.3f}, checkpoint at {args.model_path}")


def evaluate(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = resolve_data_root(args.data_root, args.dataset_id)
    model, checkpoint = load_checkpoint(resolve_model_path(args.model_path), device)

    eval_set = OrthoTargetDataset(data_root, args.split, model.image_size, augment=False)
    loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, num_workers=args.workers)
    metrics = evaluate_model(model, loader, device, top_k=args.top_k, tta=args.tta)
    logging.info(f"checkpoint from epoch {checkpoint.get('epoch')} | {_format_metrics(metrics)}")

    if args.preview_dir:
        _write_previews(model, eval_set, device, Path(args.preview_dir), args.top_k, args.tta)
        logging.info(f"previews in {args.preview_dir}")


@torch.no_grad()
def _write_previews(model, dataset, device, out_dir: Path, top_k: int, tta: bool):
    """Ground truth in green, ranked predictions in red."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        image, points, mask, _ = dataset[i]
        uv, scores = predict(model, image[None].to(device), tta=tta, top_k=top_k)
        canvas = cv2.resize(dataset.decode(i), (model.image_size, model.image_size))
        for (gx, gy) in points[mask > 0].numpy():
            cv2.drawMarker(canvas, (int(gx), int(gy)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        for rank, ((u, v), score) in enumerate(zip(uv[0].cpu().numpy(), scores[0].cpu().numpy())):
            cv2.circle(canvas, (int(u), int(v)), 10, (0, 0, 255), 2 if rank == 0 else 1)
            cv2.putText(canvas, f"{score:.2f}", (int(u) + 12, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / dataset.samples[i]["file_name"]), canvas)


def _format_metrics(metrics):
    return " ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )
