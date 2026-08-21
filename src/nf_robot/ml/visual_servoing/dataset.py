#!/usr/bin/env python

"""Loader for the mined/synthesised visual servoing dataset.

Reads the parquet shards mine_teleop.py writes: one row per frame, the frame itself in
an `image` column as JPEG bytes, and every label nullable. Null means "mask this head's
loss for this row" rather than "the answer is zero", so each item carries a mask beside
each label and the trainer never has to guess which producer wrote the row.

The JPEG bytes are held in memory, decoded per item. They are small - the frames are
stored at the model's input resolution - and holding them beats a random seek into a
shard per sample. The total is logged at load so a dataset that outgrows this is
obvious rather than mysterious.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from nf_robot.ml.ortho_target import IMAGENET_MEAN, IMAGENET_STD, photometric_jitter

# Scales that put each state component roughly in -1..1 before it reaches the FiLM MLP.
STATE_SCALES = {
    "laser_rangefinder": 1.0,   # metres; the sensor tops out near 1.3
    "finger_angle": 90.0,       # degrees, roughly -90..90
    "target_force": 1.0,        # already normalized
}
STATE_KEYS = ("laser_rangefinder", "finger_angle", "target_force")

LABEL_COLUMNS = [
    "split_source", "source_repo_id", "episode_index", "frame_index",
    "seconds_to_grasp", "target_uv", "target_range_m", "grasp_axis_rad",
    "finger", "target_present", "holding", "state",
]


def state_vector(state: dict) -> np.ndarray:
    return np.array([float(state[k]) / STATE_SCALES[k] for k in STATE_KEYS], dtype=np.float32)


class VisualServoDataset(torch.utils.data.Dataset):
    """Rows of one split, as (image, state, labels, masks)."""

    def __init__(self, root: Path, split: str, augment: bool, keep=None):
        import pyarrow.parquet as pq

        self.dir = Path(root) / split
        shards = sorted(self.dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards at {self.dir}")

        self.images: list[bytes] = []
        self.rows: list[dict] = []
        for shard in shards:
            table = pq.read_table(shard)
            images = table.column("image").to_pylist()
            labels = table.select(LABEL_COLUMNS).to_pylist()
            for blob, row in zip(images, labels):
                if keep is not None and not keep(row):
                    continue
                self.images.append(blob)
                self.rows.append(row)

        self.augment = augment
        total = sum(len(b) for b in self.images)
        logging.info(f"{self.dir}: {len(self.rows)} rows, {total / 1e6:.0f} MB of frames in memory")

    def __len__(self):
        return len(self.rows)

    def groups(self):
        """(source, episode) for every row, for splitting without leaking an episode."""
        return [(r["source_repo_id"], r["episode_index"]) for r in self.rows]

    def labelled_uv(self):
        """Every present target_uv, for the constant-prediction baseline."""
        return np.array([r["target_uv"] for r in self.rows if r["target_uv"] is not None],
                        dtype=np.float32)

    def labelled_axis(self):
        """Every present grasp_axis_rad, for the loss's angle-bin weights."""
        return np.array([r["grasp_axis_rad"] for r in self.rows
                         if r["grasp_axis_rad"] is not None], dtype=np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        bgr = cv2.imdecode(np.frombuffer(self.images[idx], np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"row {idx} has an undecodable image")
        img = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        uv = row["target_uv"]
        uv = [float(uv[0]), float(uv[1])] if uv is not None else [0.0, 0.0]
        angle = float(row["grasp_axis_rad"]) if row["grasp_axis_rad"] is not None else 0.0

        if self.augment:
            rng = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            # Horizontal flip only. A vertical flip is not a pose this camera can be in -
            # the fingers occupy specific edges and the lighting is top-biased - so the
            # full dihedral group ortho_target uses would be teaching a lie here.
            if torch.rand((), generator=rng) < 0.5:
                img = torch.flip(img, dims=(-1,))
                uv[0] = 1.0 - uv[0]
                angle = -angle
            img = photometric_jitter(img, rng)

        img = (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)

        has_uv = row["target_uv"] is not None
        return {
            "image": img,
            "state": torch.from_numpy(state_vector(row["state"])),
            "target_uv": torch.tensor(uv, dtype=torch.float32),
            "target_range_m": torch.tensor(float(row["target_range_m"] or 0.0)),
            "grasp_axis_rad": torch.tensor(angle, dtype=torch.float32),
            "finger": torch.tensor(float(row["finger"] or 0.0)),
            "present": torch.tensor(float(row["target_present"] or 0.0)),
            "holding": torch.tensor(float(row["holding"] or 0.0)),
            "has_uv": torch.tensor(float(has_uv)),
            "has_axis": torch.tensor(float(row["grasp_axis_rad"] is not None)),
            "has_finger": torch.tensor(float(row["finger"] is not None)),
            "has_present": torch.tensor(float(row["target_present"] is not None)),
            "has_holding": torch.tensor(float(row["holding"] is not None)),
        }
