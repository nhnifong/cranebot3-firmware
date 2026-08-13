#!/usr/bin/env python

"""Predict where the robot should reach next, in the ortho floor view's pixel space.

A separate approach to nf_robot.ml.target_heatmap, which learns from hand-labelled
anchor camera frames. Here the labels come from teleop recordings instead: wherever
an operator actually grasped something is by construction a place worth reaching
for, so every episode donates one label for free.

Build, distill, train, then ship. The first two are separate because the intermediate
is a LeRobot dataset and the result is not:

  1. The two recipes merge the teleop datasets, keeping the ortho feed this model
     needs and the gripper feed visual_servoing/mine_teleop.py needs, and run contact
     labelling. One dataset serves both models because the expensive part - sourcing,
     excluding episodes and re-encoding video - is identical for each. They are split
     by room so that a whole room can be held out; see the note on the eval split
     below. Build both:

       python src/nf_robot/ml/lerobot_build_dataset.py \
           --recipe src/nf_robot/ml/recipes/combined_targets.yaml \
           --temp_dir /home/nhn/data_scratch \
           --output_root /home/nhn/data_scratch/combined_targets

       python src/nf_robot/ml/lerobot_build_dataset.py \
           --recipe src/nf_robot/ml/recipes/combined_targets_eval.yaml \
           --temp_dir /home/nhn/data_scratch \
           --output_root /home/nhn/data_scratch/combined_targets_eval

  2. `distill` reduces that to one sample per episode - the episode's first ortho
     frame and the ortho pixel where contact eventually happened - which is a few
     hundred MB rather than a few hundred GB. One run per split, and --upload needs
     both splits present locally because it prunes hub files that are absent - so
     distill the eval split first, or upload only on the second run:

       python -m nf_robot.ml.ortho_target distill \
           --repo_id naavox/combined_targets_eval \
           --root /home/nhn/data_scratch/combined_targets_eval --split eval

       python -m nf_robot.ml.ortho_target distill \
           --repo_id naavox/combined_targets \
           --root /home/nhn/data_scratch/combined_targets --split train --upload

  3. `train` fits the model, saving the best checkpoint by recall@20cm to
     models/ortho_target.pth:

       python -m nf_robot.ml.ortho_target train

  4. `evaluate` scores that checkpoint against the held-out room, and --preview_dir
     draws what it actually predicted: the label in green, the ranked candidates in
     red. Numbers say whether it is right, the previews say whether it is right for
     the right reason, which is the check worth doing before a model reaches a robot:

       python -m nf_robot.ml.ortho_target evaluate --tta --preview_dir previews

  5. Try it on a robot before publishing. --local_models makes the observer load
     models/ortho_target.pth - where training just wrote it - instead of the hub
     copy. Targeting is switched on from the UI, which asks for this model by
     default (see TargetModelAction in protos/control.proto):

       stringman-headless --local_models

  6. Publish for stringman users, into the same hub repo the heatmap model lives in.
     Until this is done, anyone without --local_models is still on the old model:

       hf upload naavox/targeting models/ortho_target.pth ortho_target.pth

The ortho view is an orthographic projection of the floor plane (host/floor_view.py),
so room metres map to its pixels analytically - no camera pose is involved, unlike
camera_goal.py's per-anchor projection.

The eval split is a whole held-out room, not a random sample of episodes. Consecutive
episodes in one recording session share a floor and usually most of an object layout
- the operator clearing one pile item by item - so a random split scores near
duplicates and says nothing about a room the model has not seen. Splitting the sources
across two recipes is what makes that possible: the merge renumbers episodes and
discards which source each came from, so by the time a sample exists it is too late to
separate the rooms. combined_targets_eval.yaml is what does it: the 79west room and
nothing else, with every processing setting identical to combined_targets.yaml so the
two differ in which room they hold and in nothing else.

Caveats worth knowing before trusting the labels:

  - The projection assumes contact happens on the floor plane. An object grasped at
    height z appears displaced in the ortho view by the anchor cameras' parallax, so
    the label drifts outward from the room centre as z grows. contact_m carries the
    full 3D position, so samples can be filtered on z later.
  - Contact position is recomputed here from observation.state rather than read from
    the labelled contact_vec_* action components. Both use the same definition (see
    contact_blend_alphas), but an episode that never reaches the pressure threshold
    gets a zeroed contact_vec, which is indistinguishable from "contact at the
    starting position". Reading the pressure directly is what makes those episodes
    skippable instead of silently mislabelled.
"""

import argparse
import json
import logging
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from nf_robot.ml.lerobot_label_contact_actions import contact_blend_alphas
from nf_robot.ml.stringman_lerobot import _FEED_NAMES

ORTHO_FEED = 3
ORTHO_KEY = f"observation.images.{_FEED_NAMES[ORTHO_FEED]}"
# Both feeds this dataset carries: the ortho composite for this model, and the
# gripper camera at the visual servoing model's input size. See combined_targets.yaml.
ORTHO_CAMERA_MODE = "gripper_ortho"

# Metres of floor the square ortho map spans, per side, centred on the room origin.
# host/floor_view.py's EXTENT_M and the map_extent_meters observer._ortho_worker
# renders with; a recording made with a different extent would need its own value.
ORTHO_EXTENT_M = 5.0

DEFAULT_SOURCE_REPO_ID = "naavox/combined_targets"
DEFAULT_DATASET_ID = "naavox/ortho-target-dataset"
LOCAL_DATASET_ROOT = "ortho_target_data"

STATE_COMPONENTS = ("gripper_pos_x", "gripper_pos_y", "gripper_pos_z", "finger_pressure")


def room_to_ortho_px(x_m, y_m, width, height, extent_m=ORTHO_EXTENT_M):
    """Room-frame metres -> ortho pixel coordinates.

    Mirrors the M matrix in floor_view.generate_orthographic_floor_maps: the room
    origin is the image centre, x grows rightward, y grows upward (so v is flipped),
    and the image covers extent_m per side whatever resolution it was stored at.
    """
    u = x_m * (width / extent_m) + width / 2.0
    v = -y_m * (height / extent_m) + height / 2.0
    return u, v


def ortho_px_to_room(u, v, width, height, extent_m=ORTHO_EXTENT_M):
    """Ortho pixel coordinates -> room-frame metres; the inverse of room_to_ortho_px."""
    x_m = (u - width / 2.0) * extent_m / width
    y_m = -(v - height / 2.0) * extent_m / height
    return x_m, y_m


def frame_to_bgr(frame):
    """A LeRobot video frame (CHW float RGB, or HWC uint8) as an HWC uint8 BGR array."""
    arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def coverage_fraction(bgr):
    """Fraction of pixels any camera actually painted.

    generate_orthographic_floor_maps leaves uncovered floor at zero, so a frame
    recorded before the anchor cameras came up is almost entirely black and carries
    no scene for a model to key on.
    """
    return float(np.count_nonzero(bgr.any(axis=2))) / float(bgr.shape[0] * bgr.shape[1])


def scan_episode_states(root: Path) -> dict[int, list[dict]]:
    """Per-episode gripper position, pressure and timestamp, ordered by frame index.

    Read straight from the parquets rather than through LeRobotDataset: only four of
    the 44 state components are wanted, for every frame of every episode, and pulling
    them as columns skips decoding a video frame per row.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    index_of = {name: i for i, name in enumerate(names)}

    missing = [name for name in STATE_COMPONENTS if name not in index_of]
    if missing:
        raise ValueError(
            f"{root} has no observation.state component(s) {missing}; the contact "
            f"position cannot be recovered without them. Present: {names}"
        )
    pos_idx = [index_of[f"gripper_pos_{axis}"] for axis in "xyz"]
    pressure_idx = index_of["finger_pressure"]

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files found under {root}/data")

    episodes: dict[int, list[dict]] = {}
    for path in data_files:
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "timestamp", "observation.state"]
        )
        for ep, frame_index, timestamp, state in zip(
            table.column("episode_index").to_pylist(),
            table.column("frame_index").to_pylist(),
            table.column("timestamp").to_pylist(),
            table.column("observation.state").to_pylist(),
        ):
            episodes.setdefault(ep, []).append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "gripper_pos": [state[i] for i in pos_idx],
                "pressure": state[pressure_idx],
            })

    for rows in episodes.values():
        rows.sort(key=lambda r: r["frame_index"])
    return episodes


def episode_starts(dataset) -> dict[int, dict]:
    """Each episode's global start index, length and task string."""
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError(f"{dataset.repo_id} has no episode metadata loaded")
    columns = set(episodes.column_names)
    required = {"episode_index", "dataset_from_index", "length"}
    if not required <= columns:
        raise ValueError(
            f"episode metadata is missing {sorted(required - columns)}; this tool needs "
            f"LeRobot v3 episode metadata. Present: {sorted(columns)}"
        )

    tasks = episodes["tasks"] if "tasks" in columns else [None] * len(episodes["episode_index"])
    out = {}
    for ep, start, length, task in zip(
        episodes["episode_index"], episodes["dataset_from_index"], episodes["length"], tasks
    ):
        if isinstance(task, list):
            task = task[0] if task else None
        out[int(ep)] = {"start": int(start), "length": int(length), "task": task}
    return out


def annotate(bgr, u, v):
    """Copy of the frame with the label drawn on it, for eyeballing the projection."""
    out = bgr.copy()
    x, y = int(round(u)), int(round(v))
    cv2.drawMarker(out, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(out, (x, y), 12, (0, 255, 0), 2)
    return out


def build_samples(dataset, pressure_threshold, frame_offset, min_coverage, limit, annotate_dir):
    """One sample per episode: its early ortho frame and the contact pixel.

    Returns (samples, images, skipped) where `images` maps file name -> BGR frame and
    `skipped` counts why episodes were dropped.
    """
    root = Path(dataset.root)
    states = scan_episode_states(root)
    starts = episode_starts(dataset)

    if annotate_dir:
        Path(annotate_dir).mkdir(parents=True, exist_ok=True)

    samples, images = [], {}
    skipped = {"no_contact": 0, "too_short": 0, "blank_frame": 0, "off_map": 0}

    for ep in sorted(states):
        if limit and len(samples) >= limit:
            break
        rows = states[ep]
        meta = starts.get(ep)
        if meta is None:
            raise ValueError(f"episode {ep} has frames but no episode metadata")

        contact_index, _ = contact_blend_alphas(
            [r["timestamp"] for r in rows], [r["pressure"] for r in rows],
            pressure_threshold, blend_seconds=0.0,
        )
        if contact_index is None:
            skipped["no_contact"] += 1
            continue
        if frame_offset >= len(rows):
            skipped["too_short"] += 1
            continue

        contact = rows[contact_index]
        bgr = frame_to_bgr(dataset[meta["start"] + frame_offset][ORTHO_KEY])
        height, width = bgr.shape[:2]

        if coverage_fraction(bgr) < min_coverage:
            skipped["blank_frame"] += 1
            continue

        x_m, y_m, z_m = contact["gripper_pos"]
        u, v = room_to_ortho_px(x_m, y_m, width, height)
        if not (0 <= u < width and 0 <= v < height):
            skipped["off_map"] += 1
            continue

        file_name = f"ep{ep:06d}.jpg"
        images[file_name] = bgr
        samples.append({
            "file_name": file_name,
            # "points" (a list, of one) rather than a bare pair, so this dataset has
            # the same row shape as the hand-labelled target_heatmap one
            "points": [[round(u, 2), round(v, 2)]],
            "contact_m": [round(c, 4) for c in (x_m, y_m, z_m)],
            "episode_index": ep,
            "contact_frame_index": contact["frame_index"],
            "contact_time_s": round(contact["timestamp"], 3),
            "task": meta["task"],
        })

        if annotate_dir:
            cv2.imwrite(os.path.join(annotate_dir, file_name), annotate(bgr, u, v))

    return samples, images, skipped


def write_split(output_root: Path, split: str, samples, images) -> int:
    """Replace one split of the image folder, leaving the other one alone.

    train and eval come from separate recipes and so from separate distill runs, which
    is what keeps a whole room out of training; each run owns its own split directory.
    """
    split_dir = output_root / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    with open(split_dir / "metadata.jsonl", "w") as f:
        for sample in sorted(samples, key=lambda s: s["file_name"]):
            cv2.imwrite(str(split_dir / sample["file_name"]), images[sample["file_name"]])
            f.write(json.dumps(sample) + "\n")

    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for name, dirname in (("train", "train"), ("test", "eval")):
        if (output_root / dirname / "metadata.jsonl").exists():
            lines += [f"  - split: {name}", f"    path: {dirname}/metadata.jsonl"]
    (output_root / "README.md").write_text("\n".join(lines) + "\n---\n")
    return len(samples)


def upload_dataset(output_root: Path, dataset_id: str):
    """Replace the hub copy with this one.

    Every sample here is derived from the teleop datasets by a deterministic rule, so
    a full replacement loses nothing that cannot be regenerated - and pruning is what
    keeps samples of episodes since dropped by a recipe out of the dataset.
    """
    from huggingface_hub import HfApi, create_repo

    missing = [s for s in ("train", "eval") if not (output_root / s / "metadata.jsonl").exists()]
    if missing:
        raise ValueError(
            f"{output_root} has no {missing} split yet. The upload prunes hub files that are "
            f"absent locally, so uploading now would delete the other split's images. Distill "
            f"both recipes into this directory first."
        )

    create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    HfApi().upload_folder(
        folder_path=str(output_root),
        repo_id=dataset_id,
        repo_type="dataset",
        delete_patterns=["*.jpg"],
    )
    logging.info(f"Uploaded to {dataset_id}")


def distill(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root) if args.root else None
    dataset = LeRobotDataset(repo_id=args.repo_id, root=root)
    if ORTHO_KEY not in dataset.meta.video_keys:
        raise ValueError(
            f"'{args.repo_id}' has no '{ORTHO_KEY}' feature, so it carries no ortho view. "
            f"Build it with recipes/ortho_target.yaml. Present: {list(dataset.meta.video_keys)}"
        )

    logging.info(
        f"Distilling {dataset.meta.total_episodes} episode(s) of '{args.repo_id}' "
        f"({dataset.meta.total_frames} frames)"
    )
    samples, images, skipped = build_samples(
        dataset,
        pressure_threshold=args.pressure_threshold,
        frame_offset=args.frame_offset,
        min_coverage=args.min_coverage,
        limit=args.limit,
        annotate_dir=args.annotate_dir,
    )
    if not samples:
        raise ValueError(f"No usable episodes found. Skipped: {skipped}")

    output_root = Path(args.output)
    written = write_split(output_root, args.split, samples, images)

    logging.info(f"Wrote {written} sample(s) to {output_root / args.split}")
    dropped = ", ".join(f"{reason}={n}" for reason, n in skipped.items() if n)
    logging.info(f"Dropped episodes: {dropped or 'none'}")
    if args.annotate_dir:
        logging.info(f"Annotated previews in {args.annotate_dir}")

    if args.upload:
        upload_dataset(output_root, args.dataset_id)
    else:
        logging.info(f"Not uploading; pass --upload to push to {args.dataset_id}")


# ==========================================
# AUGMENTATION
# ==========================================
# The ortho view is a metric top-down map and gravity points out of the image, so the
# floor has no canonical orientation: any of the 8 ways of turning a square onto
# itself (4 rotations x optionally mirrored) produces an image that could equally have
# been recorded. Those 8 are exact - whole-pixel moves, no resampling and no
# interpolation blur - and the label moves with the image by the same rule, so this is
# an 8x dataset with no added label noise. Nothing else in the augmentation set is
# free like this; it is the main defence against 1400 samples of memorised floor.


def dihedral_image(img, t: int):
    """One of the 8 square symmetries applied to a CHW (or BCHW) tensor."""
    out = torch.rot90(img, t % 4, dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if t >= 4 else out


def dihedral_point(u, v, t: int, size: int):
    """The same symmetry applied to a point, in pixel-centre coordinates."""
    last = size - 1
    for _ in range(t % 4):
        u, v = v, last - u  # rot90 on dims (-2, -1) is counter-clockwise
    if t >= 4:
        u = last - u
    return u, v


def inverse_dihedral_map(m, t: int):
    """Undo dihedral_image on a spatial map, so predictions can be averaged."""
    if t >= 4:
        m = torch.flip(m, dims=(-1,))
    return torch.rot90(m, -(t % 4), dims=(-2, -1))


def photometric_jitter(img, rng: torch.Generator):
    """Brightness/contrast/saturation jitter, plus an occasional drop to grayscale.

    Floor colour is the shortcut this model must not take, so the augmentation that
    matters most is the one that makes colour unreliable.
    """
    def uniform(lo, hi):
        return torch.empty((), device=img.device).uniform_(lo, hi, generator=rng)

    img = img * uniform(0.7, 1.3)                                  # brightness
    mean = img.mean(dim=(-2, -1), keepdim=True)
    img = (img - mean) * uniform(0.7, 1.3) + mean                  # contrast
    gray = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)).sum(0, keepdim=True)
    if torch.rand((), device=img.device, generator=rng) < 0.15:
        img = gray.expand_as(img).clone()
    else:
        img = gray + (img - gray) * uniform(0.6, 1.4)              # saturation
    return img.clamp(0.0, 1.0)


# ==========================================
# DATASET
# ==========================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OrthoTargetDataset(torch.utils.data.Dataset):
    """Distilled ortho frames and their contact point, as (image, uv) pairs."""

    def __init__(self, root: Path, split: str, image_size: int, augment: bool, seed: int = 0):
        self.dir = Path(root) / split
        meta_path = self.dir / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No {split} split at {meta_path}")
        self.samples = [json.loads(line) for line in open(meta_path) if line.strip()]
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def stored_size(self):
        probe = cv2.imread(str(self.dir / self.samples[0]["file_name"]))
        return probe.shape[1], probe.shape[0]

    def scaled_labels(self):
        """Every label in the model's pixel space, without decoding the images."""
        width, height = self.stored_size()
        scale = np.array([self.image_size / width, self.image_size / height])
        return np.array([s["points"][0] for s in self.samples], dtype=np.float64) * scale

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bgr = cv2.imread(str(self.dir / sample["file_name"]))
        if bgr is None:
            raise ValueError(f"Failed to load {self.dir / sample['file_name']}")

        (u, v), = sample["points"]
        h, w = bgr.shape[:2]
        if (w, h) != (self.image_size, self.image_size):
            u *= self.image_size / w
            v *= self.image_size / h
            bgr = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if self.augment:
            # Seeded per (epoch-agnostic) index draw so workers don't share a stream.
            rng = torch.Generator().manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
            t = int(torch.randint(0, 8, (1,), generator=rng).item())
            img = dihedral_image(img, t)
            u, v = dihedral_point(u, v, t, self.image_size)
            img = photometric_jitter(img, rng)

        img = (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return img, torch.tensor([u, v], dtype=torch.float32)


# ==========================================
# MODEL
# ==========================================

DEFAULT_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_GRID = 128
DEFAULT_MODEL_PATH = "models/ortho_target.pth"


class OrthoTargetNet(nn.Module):
    """Frozen DINOv3 patch features -> one softmax over floor locations.

    The output is a single categorical distribution over grid x grid cells rather
    than a per-pixel sigmoid heatmap. Several objects on the floor are all plausible
    next targets and the operator picked one, so the honest output is a distribution
    with a mode per candidate; a softmax over locations gives that, needs no
    threshold to decode, and spends none of its loss on the 99.8% of pixels that are
    background. The offset head then places the point inside the winning cell, which
    is what keeps precision finer than the cell size.
    """

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 grid=DEFAULT_GRID, fuse_layers=4, width=256, freeze=True):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone_id)
        self.backbone_id = backbone_id
        self.image_size = image_size
        self.grid = grid
        self.fuse_layers = fuse_layers
        self.freeze = freeze

        config = self.backbone.config
        self.patch_size = config.patch_size
        self.token_grid = image_size // self.patch_size
        if image_size % self.patch_size:
            raise ValueError(f"image_size {image_size} is not a multiple of patch {self.patch_size}")

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        in_ch = config.hidden_size * fuse_layers
        # Two bilinear x2 steps take the 16px token grid to 4px cells at 512 input.
        ups = int(math.log2(grid / self.token_grid))
        if 2 ** ups * self.token_grid != grid:
            raise ValueError(f"grid {grid} is not a power-of-two multiple of {self.token_grid}")

        layers = [nn.Conv2d(in_ch, width, 1), nn.GroupNorm(32, width), nn.GELU()]
        channels = width
        for _ in range(ups):
            nxt = max(64, channels // 2)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, nxt, 3, padding=1), nn.GroupNorm(32, nxt), nn.GELU(),
            ]
            channels = nxt
        self.decoder = nn.Sequential(*layers)
        self.logit_head = nn.Conv2d(channels, 1, 1)
        self.offset_head = nn.Conv2d(channels, 2, 1)

    def features(self, pixel_values):
        with torch.set_grad_enabled(self.training and not self.freeze):
            out = self.backbone(pixel_values, output_hidden_states=True)
        n_patches = self.token_grid ** 2
        # [CLS] and the register tokens lead the sequence; patches are always the tail.
        feats = [h[:, -n_patches:, :] for h in out.hidden_states[-self.fuse_layers:]]
        x = torch.cat(feats, dim=-1).transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], self.token_grid, self.token_grid)

    def forward(self, pixel_values):
        x = self.decoder(self.features(pixel_values))
        return self.logit_head(x).squeeze(1), self.offset_head(x)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # a frozen backbone must not update its norm statistics
        return self


def location_loss(logits, offsets, target_uv, image_size, grid, offset_weight=1.0):
    """Cross-entropy over cells, plus L1 on the sub-cell offset of the true cell."""
    scale = image_size / grid
    cell = target_uv / scale
    cx = cell[:, 0].floor().clamp(0, grid - 1).long()
    cy = cell[:, 1].floor().clamp(0, grid - 1).long()
    index = cy * grid + cx

    ce = F.cross_entropy(logits.flatten(1), index)

    frac = cell - torch.stack([cx, cy], dim=1).float()
    picked = offsets.flatten(2).gather(2, index[:, None, None].expand(-1, 2, -1)).squeeze(-1)
    l1 = F.l1_loss(picked.sigmoid(), frac.clamp(0.0, 1.0))
    return ce + offset_weight * l1, ce.detach(), l1.detach()


def decode(logits, offsets, image_size, grid, top_k=1, nms_radius=2):
    """Peak cells plus their offsets, as (B, k, 2) pixel coordinates and (B, k) scores.

    Peaks, not the expectation over the map: averaging two candidate objects would
    land the prediction on the empty floor between them.
    """
    scale = image_size / grid
    prob = logits.flatten(1).softmax(1).view(-1, 1, grid, grid)

    if top_k > 1:
        # Suppress everything that is not a local maximum so the k results are k
        # distinct candidates rather than k cells of one blob.
        pooled = F.max_pool2d(prob, nms_radius * 2 + 1, stride=1, padding=nms_radius)
        prob = torch.where(prob >= pooled, prob, torch.zeros_like(prob))

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    cx = (index % grid).float()
    cy = torch.div(index, grid, rounding_mode="floor").float()

    off = offsets.flatten(2).sigmoid()
    picked = torch.stack([
        off[:, 0].gather(1, index),
        off[:, 1].gather(1, index),
    ], dim=-1)
    uv = (torch.stack([cx, cy], dim=-1) + picked) * scale
    return uv, scores


def predict(model, images, tta=False, top_k=1):
    """Model outputs for a batch, optionally averaged over the 8 square symmetries."""
    logits, offsets = model(images)
    if tta:
        acc = torch.zeros_like(logits)
        for t in range(8):
            transformed, _ = model(dihedral_image(images, t))
            prob = transformed.flatten(1).softmax(1).view_as(transformed)
            acc += inverse_dihedral_map(prob, t)
        # Averaged in probability space; back to logits so decode can treat it uniformly.
        # Offsets stay from the untransformed pass, being sub-cell either way.
        logits = (acc / 8).clamp_min(1e-12).log()
    return decode(logits, offsets, model.image_size, model.grid, top_k=top_k)


# ==========================================
# LIVE INFERENCE
# ==========================================
# The observer's ortho worker renders a larger map than the recordings stored, so the
# only step between the live frame and the model is the same resize the training
# loader applies. Room metres come back out analytically, no camera pose involved.

TARGETING_MODEL_REPOID = "naavox/targeting"
TARGETING_MODEL_FILENAME = "ortho_target.pth"


def prepare_ortho_image(rgb, image_size, device):
    """A live ortho frame as a one-image normalized batch, as the training loader made them.

    The frame is RGB, not BGR: anchor clients decode to rgb24 and the floor projection
    keeps that order, which is also the order that reached training - the recorded
    videos are converted to BGR only on their way into the video streamer.
    """
    if rgb.shape[:2] != (image_size, image_size):
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    img = (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return img[None].to(device)


@torch.no_grad()
def predict_room_targets(model, rgb, device, top_k=1, tta=False):
    """Best target(s) for one ortho frame, as [(x_m, y_m, score)] in the room frame."""
    batch = prepare_ortho_image(rgb, model.image_size, device)
    uv, scores = predict(model, batch, tta=tta, top_k=top_k)
    out = []
    for (u, v), score in zip(uv[0].cpu().numpy(), scores[0].cpu().numpy()):
        x_m, y_m = ortho_px_to_room(float(u), float(v), model.image_size, model.image_size)
        out.append((x_m, y_m, float(score)))
    return out


# ==========================================
# TRAINING
# ==========================================


def pixels_to_cm(pixels, image_size):
    """Ortho pixels to centimetres of floor, using the map's fixed metres per side."""
    return pixels * (ORTHO_EXTENT_M * 100.0) / image_size


@torch.no_grad()
def evaluate_model(model, loader, device, top_k=5, tta=False, radii_cm=(10, 20, 50)):
    """Distance error and hit rates, plus the top-k rate that measures 'picked something'."""
    model.eval()
    errors, best_of_k, nll, count = [], [], 0.0, 0
    for images, target_uv in loader:
        images, target_uv = images.to(device), target_uv.to(device)
        logits, offsets = model(images)
        _, ce, _ = location_loss(logits, offsets, target_uv, model.image_size, model.grid)
        nll += ce.item() * images.shape[0]
        count += images.shape[0]

        uv, _ = predict(model, images, tta=tta, top_k=top_k)
        distance = (uv - target_uv[:, None, :]).norm(dim=-1)
        errors.append(distance[:, 0].cpu())
        best_of_k.append(distance.min(dim=1).values.cpu())

    errors = pixels_to_cm(torch.cat(errors), model.image_size)
    best_of_k = pixels_to_cm(torch.cat(best_of_k), model.image_size)
    metrics = {
        "nll": nll / max(count, 1),
        "median_cm": errors.median().item(),
        "mean_cm": errors.mean().item(),
    }
    for radius in radii_cm:
        metrics[f"recall@{radius}cm"] = (errors <= radius).float().mean().item()
    metrics[f"top{top_k}@20cm"] = (best_of_k <= 20).float().mean().item()
    return metrics


def constant_baseline(train_set, eval_set, image_size, radii_cm=(10, 20, 50)):
    """Score of always predicting the mean training label.

    A floor-target prior is strong enough that this is not a trivial number, and any
    model that fails to beat it has learned nothing about the image.
    """
    train_uv = train_set.scaled_labels()
    eval_uv = eval_set.scaled_labels()
    errors = pixels_to_cm(np.linalg.norm(eval_uv - train_uv.mean(0), axis=1), image_size)
    out = {"median_cm": float(np.median(errors)), "mean_cm": float(errors.mean())}
    for radius in radii_cm:
        out[f"recall@{radius}cm"] = float((errors <= radius).mean())
    return out


def resolve_data_root(args) -> Path:
    """Local distilled dataset, downloading it from the hub if no local copy was named."""
    if args.data_root:
        return Path(args.data_root)
    from huggingface_hub import snapshot_download

    logging.info(f"Downloading {args.dataset_id}")
    return Path(snapshot_download(repo_id=args.dataset_id, repo_type="dataset"))


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    data_root = resolve_data_root(args)
    train_set = OrthoTargetDataset(data_root, "train", args.image_size, augment=True, seed=args.seed)
    eval_set = OrthoTargetDataset(data_root, "eval", args.image_size, augment=False)
    logging.info(f"train {len(train_set)} sample(s) | eval {len(eval_set)} sample(s) from {data_root}")

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
    ).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if args.unfreeze_backbone:
        groups.append({"params": list(model.backbone.parameters()), "lr": args.lr * args.backbone_lr_scale})
        logging.info(f"backbone unfrozen at {args.backbone_lr_scale}x the head learning rate")
    else:
        logging.info(f"backbone frozen; training {sum(p.numel() for p in head_params) / 1e6:.1f}M head parameters")

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = max(1, len(train_loader)) * args.epochs
    warmup = max(1, int(0.05 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    ))
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(3)
        for images, target_uv in train_loader:
            images, target_uv = images.to(device, non_blocking=True), target_uv.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits, offsets = model(images)
                loss, ce, l1 = location_loss(
                    logits.float(), offsets.float(), target_uv,
                    args.image_size, args.grid, args.offset_weight,
                )
            loss.backward()
            optimizer.step()
            schedule.step()
            totals += [loss.item(), ce.item(), l1.item()]

        totals /= max(1, len(train_loader))
        line = f"epoch {epoch + 1}/{args.epochs} loss {totals[0]:.4f} (ce {totals[1]:.4f} off {totals[2]:.4f})"

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            metrics = evaluate_model(model, eval_loader, device, top_k=args.top_k)
            logging.info(f"{line} | {_format_metrics(metrics)}")
            score = metrics["recall@20cm"]
            if score > best:
                best = score
                torch.save({
                    "state_dict": model.state_dict(),
                    "backbone_id": args.backbone,
                    "image_size": args.image_size,
                    "grid": args.grid,
                    "fuse_layers": args.fuse_layers,
                    "metrics": metrics,
                    "epoch": epoch + 1,
                }, args.model_path)
                logging.info(f"saved {args.model_path} (recall@20cm {score:.3f})")
        else:
            logging.info(line)

    logging.info(f"done; best eval recall@20cm {best:.3f}, checkpoint at {args.model_path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = OrthoTargetNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        grid=checkpoint["grid"], fuse_layers=checkpoint["fuse_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = resolve_data_root(args)
    model, checkpoint = load_checkpoint(args.model_path, device)

    eval_set = OrthoTargetDataset(data_root, args.split, model.image_size, augment=False)
    loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, num_workers=args.workers)
    metrics = evaluate_model(model, loader, device, top_k=args.top_k, tta=args.tta)
    logging.info(f"checkpoint from epoch {checkpoint.get('epoch')} | {_format_metrics(metrics)}")

    if args.preview_dir:
        _write_previews(model, eval_set, device, Path(args.preview_dir), args.top_k, args.tta)
        logging.info(f"previews in {args.preview_dir}")


@torch.no_grad()
def _write_previews(model, dataset, device, out_dir: Path, top_k: int, tta: bool):
    """Ground truth in green, ranked predictions in red, for looking at with your eyes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        image, target_uv = dataset[i]
        uv, scores = predict(model, image[None].to(device), tta=tta, top_k=top_k)
        canvas = cv2.imread(str(dataset.dir / dataset.samples[i]["file_name"]))
        canvas = cv2.resize(canvas, (model.image_size, model.image_size))
        gt = target_uv.numpy()
        cv2.drawMarker(canvas, (int(gt[0]), int(gt[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        for rank, ((u, v), score) in enumerate(zip(uv[0].cpu().numpy(), scores[0].cpu().numpy())):
            cv2.circle(canvas, (int(u), int(v)), 10, (0, 0, 255), 2 if rank == 0 else 1)
            cv2.putText(canvas, f"{score:.2f}", (int(u) + 12, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / dataset.samples[i]["file_name"]), canvas)


def _format_metrics(metrics):
    return " ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    distill_parser = subparsers.add_parser(
        "distill", help="reduce an ortho LeRobot dataset to one image + contact point per episode"
    )
    distill_parser.add_argument("--repo_id", default=DEFAULT_SOURCE_REPO_ID,
                                help="LeRobot dataset built by recipes/ortho_target.yaml")
    distill_parser.add_argument("--root", default=None,
                                help="local root of that dataset (default: the HF cache, downloading if needed)")
    distill_parser.add_argument("--output", default=LOCAL_DATASET_ROOT,
                                help="directory to write the distilled dataset into")
    distill_parser.add_argument("--split", default="train", choices=["train", "eval"],
                                help="which split this source becomes; only that directory is "
                                     "rewritten, so the two recipes can be distilled independently")
    distill_parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                                help="hub repo to upload the distilled dataset to")
    distill_parser.add_argument("--upload", action="store_true",
                                help="upload the result, replacing the hub copy")
    distill_parser.add_argument("--pressure_threshold", type=float, default=0.1,
                                help="finger_pressure above which an episode counts as having made contact")
    distill_parser.add_argument("--frame_offset", type=int, default=0,
                                help="which frame of each episode to keep, counted from its start")
    distill_parser.add_argument("--min_coverage", type=float, default=0.02,
                                help="skip frames where less than this fraction of the ortho map was "
                                     "painted by any camera")
    distill_parser.add_argument("--limit", type=int, default=0,
                                help="stop after this many samples, for a quick trial run")
    distill_parser.add_argument("--annotate_dir", default=None,
                                help="also write copies with the label drawn on them, to check the projection")

    def add_data_args(sub):
        sub.add_argument("--dataset_id", default=DEFAULT_DATASET_ID, help="distilled dataset on the hub")
        sub.add_argument("--data_root", default=None,
                         help="local distilled dataset directory (default: download --dataset_id)")
        sub.add_argument("--batch_size", type=int, default=32)
        sub.add_argument("--workers", type=int, default=4)
        sub.add_argument("--top_k", type=int, default=5,
                         help="candidates to decode; the top-k hit rate is the 'picked something "
                              "plausible' measure, which single-answer error cannot capture")
        sub.add_argument("--device", default=None)
        sub.add_argument("--model_path", default=DEFAULT_MODEL_PATH)

    train_parser = subparsers.add_parser("train", help="fit the model on the distilled dataset")
    add_data_args(train_parser)
    train_parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    train_parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    train_parser.add_argument("--grid", type=int, default=DEFAULT_GRID,
                              help="output cells per side; 128 over a 5m map is 3.9cm per cell")
    train_parser.add_argument("--fuse_layers", type=int, default=4,
                              help="how many of the backbone's last blocks to concatenate")
    train_parser.add_argument("--epochs", type=int, default=60)
    # Paired with --batch_size: a batch four times larger takes a quarter as many steps,
    # so the rate rises with it (sqrt of the ratio, the usual compromise for AdamW).
    train_parser.add_argument("--lr", type=float, default=6e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.05)
    train_parser.add_argument("--offset_weight", type=float, default=1.0)
    train_parser.add_argument("--eval_every", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--unfreeze_backbone", action="store_true",
                              help="also train the backbone, at --backbone_lr_scale of the head's rate")
    train_parser.add_argument("--backbone_lr_scale", type=float, default=0.05)

    eval_parser = subparsers.add_parser("evaluate", help="score a checkpoint on a held-out split")
    add_data_args(eval_parser)
    eval_parser.add_argument("--split", default="eval", choices=["train", "eval"])
    eval_parser.add_argument("--tta", action="store_true",
                             help="average predictions over the 8 square symmetries")
    eval_parser.add_argument("--preview_dir", default=None,
                             help="write images with the label and the predictions drawn on")

    args = parser.parse_args()
    if args.command == "distill":
        distill(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
