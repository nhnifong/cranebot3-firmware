#!/usr/bin/env python

"""Loader for the drop pair shards mine_teleop.py writes; see dataset.md.

One item snapshot and the overhead view taken with it, against the drop point that
episode ended in. The release overhead frame is not read: it shows the robot standing at
the answer.

Episodes are dealt to train and eval here rather than by a separate step, because every
row of an episode shares one drop point and they have to travel together.
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from nf_robot.ml.image_input import normalize, photometric_jitter, to_tensor
from nf_robot.ml.ortho_target.model import dihedral_image, dihedral_point
from nf_robot.ml.placer.model import ITEM_SIZE, OVERHEAD_SIZE

LABEL_COLUMNS = ["source_repo_id", "episode_index", "frame_index", "task", "release_ortho_uv",
                 "pickup_ortho_uv", "release_height_m"]
EVAL_FRACTION = 0.2


def deal_episodes(groups, eval_fraction=EVAL_FRACTION, seed=0):
    """Which (source, episode) pairs are eval, drawn once so both splits agree."""
    episodes = sorted(set(groups))
    random.Random(seed).shuffle(episodes)
    return set(episodes[:int(round(len(episodes) * eval_fraction))])


class DropPairDataset(torch.utils.data.Dataset):
    """Rows of one split, as (item, overhead, target uv)."""

    def __init__(self, root: Path, split: str, augment: bool, eval_fraction=EVAL_FRACTION,
                 seed=0, pool="all"):
        import pyarrow.parquet as pq

        self.dir = Path(root) / pool
        shards = sorted(self.dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards at {self.dir}")

        rows, items, overheads = [], [], []
        for shard in shards:
            table = pq.read_table(shard)
            rows += table.select(LABEL_COLUMNS).to_pylist()
            items += table.column("image").to_pylist()
            overheads += table.column("snapshot_overhead_image").to_pylist()

        groups = [(r["source_repo_id"], r["episode_index"]) for r in rows]
        held_out = deal_episodes(groups, eval_fraction, seed)
        keep = [i for i, (row, group) in enumerate(zip(rows, groups))
                if (group in held_out) == (split == "eval") and overheads[i] is not None]
        if not keep:
            raise ValueError(f"no {split} rows in {self.dir}")

        self.rows = [rows[i] for i in keep]
        self.items = [items[i] for i in keep]
        self.overheads = [overheads[i] for i in keep]
        self.augment = augment
        dropped = len(rows) - sum(o is not None for o in overheads)
        logging.info(f"{self.dir} {split}: {len(self.rows)} row(s) from "
                     f"{len({groups[i] for i in keep})} episode(s)"
                     + (f"; {dropped} row(s) have no overhead frame" if dropped else ""))

    def __len__(self):
        return len(self.rows)

    def labelled_uv(self):
        """Every drop point, for the constant-prediction baseline."""
        return np.array([r["release_ortho_uv"] for r in self.rows], dtype=np.float32)

    def _decode(self, blob, size):
        bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if (bgr.shape[1], bgr.shape[0]) != tuple(size):
            bgr = cv2.resize(bgr, tuple(size), interpolation=cv2.INTER_AREA)
        return to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = self._decode(self.items[idx], ITEM_SIZE)
        overhead = self._decode(self.overheads[idx], OVERHEAD_SIZE)
        uv = np.asarray(row["release_ortho_uv"], dtype=np.float32)

        if self.augment:
            rng = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            # The floor has no canonical orientation, so all 8 symmetries of the overhead
            # square are rooms the robot could have been in. The item is photographed by a
            # camera that hangs one way up, so it only gets the mirror.
            t = int(torch.randint(0, 8, (1,), generator=rng).item())
            overhead = dihedral_image(overhead, t)
            size = overhead.shape[-1]
            u, v = dihedral_point(uv[0] * size, uv[1] * size, t, size)
            uv = np.array([u / size, v / size], dtype=np.float32)
            if torch.rand((), generator=rng) < 0.5:
                item = torch.flip(item, dims=(-1,))
            item = photometric_jitter(item, rng)
            overhead = photometric_jitter(overhead, rng)

        return {
            "item": normalize(item),
            "overhead": normalize(overhead),
            "target_uv": torch.from_numpy(uv),
            "task": row["task"],
        }
