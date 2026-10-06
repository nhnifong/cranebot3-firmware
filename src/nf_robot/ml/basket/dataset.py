#!/usr/bin/env python

"""Loader for the basket centering shards that basket.mine writes."""

import numpy as np
import torch

from nf_robot.ml.image_input import normalize, photometric_jitter
from nf_robot.ml.shard_dataset import ShardDataset
from nf_robot.ml.visual_servoing.dataset import state_vector

LABEL_COLUMNS = [
    "split_source", "source_repo_id", "episode_index", "frame_index", "seconds_from_center",
    "target_uv", "target_range_m", "return_body", "state",
]


class BasketDataset(ShardDataset):
    """Rows of one split, as (image, state, labels)."""

    LABEL_COLUMNS = LABEL_COLUMNS
    SPLIT_COMMAND = ("python -m nf_robot.ml.basket.mine --output_root {root} ... "
                     "(mining deals the pool itself)")

    def labelled_return(self):
        """Every row's move back to centre, for the stay-put baseline."""
        return np.array([r["return_body"] for r in self.rows], dtype=np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = self.image(idx)
        uv = [float(c) for c in row["target_uv"]]
        move = [float(c) for c in row["return_body"]]

        if self.augment:
            rng = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            # Horizontal flip only: the camera is never upside down. The camera sits on the
            # body's x = 0 plane and its x axis is the body's, so the mirror image is the
            # same scene with sideways negated.
            if torch.rand((), generator=rng) < 0.5:
                img = torch.flip(img, dims=(-1,))
                uv[0] = 1.0 - uv[0]
                move[0] = -move[0]
            img = photometric_jitter(img, rng)

        return {
            "image": normalize(img),
            "state": torch.from_numpy(state_vector(row["state"])),
            "target_uv": torch.tensor(uv, dtype=torch.float32),
            "target_range_m": torch.tensor(float(row["target_range_m"])),
            "return_body": torch.tensor(move, dtype=torch.float32),
            "has_uv": torch.tensor(1.0),
        }
