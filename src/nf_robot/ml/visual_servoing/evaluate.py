#!/usr/bin/env python

"""Score a trained visual servoing checkpoint against a held-out split.

Numbers say whether the model is right; `--preview_dir` says whether it is right for the
right reason, by drawing the label as a green cross and the prediction as a red one on the
frames it was scored on. That is the check worth doing before a model reaches a robot.

What the numbers mean, all measured on the model's own 448x256 input, whose diagonal is
516 px - so an error of 112 px is about a quarter of the frame away:

    median_px     typical distance between the predicted grasp point and the true one.
                  Lower is better. The median rather than the mean because a handful of
                  frames where the model points at the wrong object dominate an average.
    mean_px       the same with those frames included, so mean >> median means a few
                  confident mistakes rather than uniform vagueness.
    recall@Npx    the fraction of frames landing within N pixels of the truth. Higher is
                  better, 1.0 is perfect. "Recall" here is a hit rate at a tolerance, not
                  the precision/recall of a detector - there is nothing to miss, since
                  every scored frame has exactly one target. Read the three radii as
                  usable (10px), close (25px) and roughly right (50px).
    range_ratio   median predicted distance over true distance. 1.0 is perfect, above 1
                  means the model thinks the floor is further away than it is.
    axis_deg      median grasp axis error in degrees. Only meaningful if the labels
                  themselves vary - a split whose axis labels are all one value can be
                  scored at a degree or two by a model that has learned to answer that
                  value and nothing else.
    finger_mae    mean error of the commanded finger speed, in the -1..1 the head predicts.
    present_acc,  accuracy of the two classifier heads. present_acc is near 1.0 for free
    holding_acc   on any split that is mostly frames with a target in them.

The constant-prediction baseline is printed first, and it is the number that matters most:
it is the score of ignoring the image entirely and always answering the average training
target. For a centering task that is a strong answer. A model that does not clearly beat
it on median_px *and* on the recall radii has learned nothing about the image, whatever
its loss curve did.

Usage:
    python -m nf_robot.ml.visual_servoing.evaluate \
        --data_root datasets/visual_servoing --model_path models/visual_servo.pth \
        --preview_dir datasets/visual_servoing/eval_predictions
    python -m nf_robot.ml.visual_servoing.evaluate \
        --data_root datasets/visual_servoing --model_path models/visual_servo.pth --upload
"""

import argparse
import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from nf_robot.ml.visual_servoing.dataset import VisualServoDataset
from nf_robot.ml.visual_servoing.model import decode, load_checkpoint
from nf_robot.ml.visual_servoing.servo import SERVO_MODEL_REPOID
from nf_robot.ml.visual_servoing.train import (
    DEFAULT_MODEL_PATH,
    constant_baseline,
    evaluate as evaluate_metrics,
    upload_model,
    _format,
)

# Grid of the contact sheets the preview writes.
PREVIEW_COLUMNS = 4
PREVIEW_GROUP = 20


class LabelsOnly:
    """The target positions of a split, read without decoding any images."""

    def __init__(self, split_dir):
        import pyarrow as pa
        import pyarrow.parquet as pq

        shards = sorted(Path(split_dir).glob("*.parquet"))
        table = pa.concat_tables([pq.read_table(s, columns=["target_uv"]) for s in shards])
        self.uv = np.array([v for v in table.column("target_uv").to_pylist() if v is not None],
                           dtype=np.float32)

    def labelled_uv(self):
        return self.uv


def undo_normalization(tensor):
    """A normalized model input back to a BGR image."""
    from nf_robot.ml.ortho_target import IMAGENET_MEAN, IMAGENET_STD

    array = tensor.cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return cv2.cvtColor((np.clip(array, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def draw_legend(canvas):
    """Say which mark is which, on the image rather than in the docs.

    These frames get looked at on their own, days later and often by someone who did not
    run the eval, and "which one is the prediction" is the first question every time.
    """
    y = canvas.shape[0] - 9
    cv2.rectangle(canvas, (0, y - 15), (canvas.shape[1], canvas.shape[0]), (25, 25, 25), -1)

    def entry(x, colour, marker, text):
        if marker is not None:
            cv2.drawMarker(canvas, (x + 6, y - 4), colour, marker, 11, 2)
        else:
            cv2.line(canvas, (x, y - 4), (x + 12, y - 4), colour, 2)
        cv2.putText(canvas, text, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

    entry(6, (0, 255, 0), cv2.MARKER_CROSS, "label")
    entry(74, (0, 0, 255), cv2.MARKER_TILTED_CROSS, "prediction")
    entry(168, (0, 140, 255), None, "predicted axis")
    entry(280, (200, 200, 200), None, "error")
    return canvas


def draw_prediction(image, label_uv, predicted, image_size):
    """One frame with its label and the model's answer drawn on a padded canvas.

    Green upright cross is the label, red tilted cross is the prediction, the orange bar
    through it is the predicted grasp axis, and the thin grey line joins the two so the
    error is a length rather than a number to look up. draw_legend puts the same on the
    image.
    """
    width, height = image_size
    pad_x, pad_y = int(width * 0.25), int(height * 0.25)
    canvas = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x,
                                cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + width, pad_y + height), (90, 90, 90), 1)

    def point(uv):
        return int(uv[0] * width + pad_x), int(uv[1] * height + pad_y)

    if label_uv is not None:
        cv2.drawMarker(canvas, point(label_uv), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    x, y = point(predicted["uv"])
    length, angle = 24, predicted["axis"]
    cv2.line(canvas, (int(x - math.cos(angle) * length), int(y - math.sin(angle) * length)),
             (int(x + math.cos(angle) * length), int(y + math.sin(angle) * length)),
             (0, 140, 255), 2)
    cv2.drawMarker(canvas, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
    if label_uv is not None:
        cv2.line(canvas, point(label_uv), (x, y), (200, 200, 200), 1)

    lines = [
        f"err {predicted['error_px']:.0f}px" if label_uv is not None else "no label",
        f"range {predicted['range_m']:.3f}m" + (
            f" (label {predicted['label_range_m']:.3f})" if label_uv is not None else ""),
        f"axis {math.degrees(predicted['axis']):+.0f}deg  finger {predicted['finger']:+.2f}",
        f"present {predicted['present']:.2f}  holding {predicted['holding']:.2f}",
    ]
    for i, text in enumerate(lines):
        y0 = 18 + i * 18
        cv2.putText(canvas, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(canvas, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return draw_legend(canvas)


@torch.no_grad()
def write_preview(model, dataset, device, preview_dir, count, seed, image_size):
    """Annotated frames plus contact sheets for a random sample of the split."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in list(preview_dir.glob("*.jpg")) + list(preview_dir.glob("*.png")):
        old.unlink()

    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    scale = torch.tensor(image_size, dtype=torch.float32)
    cells = []
    for index in indices:
        item = dataset[index]
        batch = {k: v[None].to(device) for k, v in item.items()}
        outputs = model(batch["image"], batch["state"])
        uv, distance, angle, _ = decode(outputs, model.grid, top_k=1)

        has_uv = bool(item["has_uv"])
        label_uv = item["target_uv"].numpy() if has_uv else None
        predicted = {
            "uv": uv[0, 0].cpu().numpy(),
            "range_m": float(distance[0, 0]),
            "label_range_m": float(item["target_range_m"]),
            "axis": float(angle[0, 0]),
            "finger": float(outputs["finger"][0]),
            "present": float(outputs["present_logit"][0].sigmoid()),
            "holding": float(outputs["holding_logit"][0].sigmoid()),
            "error_px": float(((uv[0, 0].cpu() - item["target_uv"]) * scale).norm()) if has_uv else 0.0,
        }
        cell = draw_prediction(undo_normalization(item["image"]), label_uv, predicted, image_size)
        cv2.imwrite(str(preview_dir / f"row{index:06d}.jpg"), cell)
        cells.append(cell)

    for start in range(0, len(cells), PREVIEW_GROUP):
        block = cells[start:start + PREVIEW_GROUP]
        blank = np.full_like(block[0], 25)
        block = block + [blank] * (-len(block) % PREVIEW_COLUMNS)
        sheet = np.vstack([np.hstack(block[r:r + PREVIEW_COLUMNS])
                           for r in range(0, len(block), PREVIEW_COLUMNS)])
        cv2.imwrite(str(preview_dir / f"_sheet_{start // PREVIEW_GROUP + 1:02d}.png"), sheet)

    logging.info(f"wrote {len(cells)} annotated frames to {preview_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="Root holding the split to score")
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preview_dir", default=None, help="Write annotated frames here")
    parser.add_argument("--preview_count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--upload", action="store_true",
                        help="Push this checkpoint to --model_id once it has been scored")
    parser.add_argument("--model_id", default=SERVO_MODEL_REPOID,
                        help="Hub model repo to push to")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.model_path, device)
    image_size = tuple(checkpoint["image_size"])
    logging.info(f"{args.model_path}: epoch {checkpoint['epoch']}, input {image_size[0]}x{image_size[1]}")

    data_root = Path(args.data_root)
    eval_set = VisualServoDataset(data_root, args.split, augment=False)
    loader = torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")

    if (data_root / "train").exists() and args.split != "train":
        baseline = constant_baseline(LabelsOnly(data_root / "train"), eval_set, image_size)
        if baseline:
            logging.info(f"constant-prediction baseline: {_format(baseline)}")

    metrics = evaluate_metrics(model, loader, device, image_size)
    logging.info(f"{args.split}: {_format(metrics)}")

    if args.preview_dir:
        write_preview(model, eval_set, device, args.preview_dir,
                      args.preview_count, args.seed, image_size)

    if args.upload:
        # Uploading from here rather than from training means what goes to the hub is the
        # checkpoint that was just scored, and the score in the commit message is the one
        # that was measured rather than the best seen mid-run.
        upload_model(args.model_path, args.model_id, metrics)


if __name__ == "__main__":
    main()
