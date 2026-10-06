#!/usr/bin/env python

"""Loader for the visual servoing parquet shards, with a mask beside every nullable label."""

import numpy as np
import torch

from nf_robot.ml.image_input import normalize, photometric_jitter
from nf_robot.ml.shard_dataset import ShardDataset

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
    "finger", "close_now", "grasp_pressure", "target_present", "holding", "state",
]


def state_vector(state: dict) -> np.ndarray:
    return np.array([float(state[k]) / STATE_SCALES[k] for k in STATE_KEYS], dtype=np.float32)


class VisualServoDataset(ShardDataset):
    """Rows of one split, as (image, state, labels, masks)."""

    LABEL_COLUMNS = LABEL_COLUMNS

    def labelled_uv(self):
        """Every present target_uv, for the constant-prediction baseline."""
        return np.array([r["target_uv"] for r in self.rows if r["target_uv"] is not None],
                        dtype=np.float32)

    def has_close_labels(self):
        return any(r.get("close_now") is not None for r in self.rows)

    def labelled_axis(self):
        """Every present grasp_axis_rad, for the loss's angle-bin weights."""
        return np.array([r["grasp_axis_rad"] for r in self.rows
                         if r["grasp_axis_rad"] is not None], dtype=np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = self.image(idx)

        uv = row["target_uv"]
        uv = [float(uv[0]), float(uv[1])] if uv is not None else [0.0, 0.0]
        angle = float(row["grasp_axis_rad"]) if row["grasp_axis_rad"] is not None else 0.0

        if self.augment:
            rng = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            # Horizontal flip only: the camera is never upside down.
            if torch.rand((), generator=rng) < 0.5:
                img = torch.flip(img, dims=(-1,))
                uv[0] = 1.0 - uv[0]
                angle = -angle
            img = photometric_jitter(img, rng)

        img = normalize(img)

        has_uv = row["target_uv"] is not None
        return {
            "image": img,
            "state": torch.from_numpy(state_vector(row["state"])),
            "target_uv": torch.tensor(uv, dtype=torch.float32),
            "target_range_m": torch.tensor(float(row["target_range_m"] or 0.0)),
            "grasp_axis_rad": torch.tensor(angle, dtype=torch.float32),
            "finger": torch.tensor(float(row["finger"] or 0.0)),
            "close_now": torch.tensor(float(row["close_now"] or 0.0)),
            "grasp_pressure": torch.tensor(float(row["grasp_pressure"] or 0.0)),
            "present": torch.tensor(float(row["target_present"] or 0.0)),
            "holding": torch.tensor(float(row["holding"] or 0.0)),
            "has_uv": torch.tensor(float(has_uv)),
            "has_axis": torch.tensor(float(row["grasp_axis_rad"] is not None)),
            "has_finger": torch.tensor(float(row["finger"] is not None)),
            "has_close": torch.tensor(float(row["close_now"] is not None)),
            "has_pressure": torch.tensor(float(row["grasp_pressure"] is not None)),
            "has_present": torch.tensor(float(row["target_present"] is not None)),
            "has_holding": torch.tensor(float(row["holding"] is not None)),
        }
