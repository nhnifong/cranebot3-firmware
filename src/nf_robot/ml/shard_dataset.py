"""Loading a split of mined parquet shards into memory, shared by the gripper camera datasets."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from nf_robot.ml.image_input import to_tensor
from nf_robot.ml.visual_servoing.mine_teleop import POOL_SPLIT


class ShardDataset(torch.utils.data.Dataset):
    """Every row of one split: its JPEG and its label columns, a missing column read as null.

    A subclass names its LABEL_COLUMNS and the command that deals its pool into splits,
    and turns a row into tensors in __getitem__.
    """

    LABEL_COLUMNS: list = []
    SPLIT_COMMAND = "python -m nf_robot.ml.visual_servoing.split_pool --data_root {root}"

    def __init__(self, root: Path, split: str, augment: bool, keep=None):
        import pyarrow.parquet as pq

        self.dir = Path(root) / split
        shards = sorted(self.dir.glob("*.parquet"))
        if not shards:
            # Most often a pool that was built but never dealt.
            pool = Path(root) / POOL_SPLIT
            hint = (f". {pool} holds shards that have not been dealt yet: run "
                    f"`{self.SPLIT_COMMAND.format(root=root)}`"
                    if any(pool.glob("*.parquet")) else "")
            raise FileNotFoundError(f"No parquet shards at {self.dir}{hint}")

        self.images: list[bytes] = []
        self.rows: list[dict] = []
        for shard in shards:
            table = pq.read_table(shard)
            images = table.column("image").to_pylist()
            # A shard without a label column means that label is null.
            present = [c for c in self.LABEL_COLUMNS if c in table.schema.names]
            missing = [c for c in self.LABEL_COLUMNS if c not in present]
            labels = table.select(present).to_pylist()
            if missing:
                for row in labels:
                    row.update(dict.fromkeys(missing))
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

    def image(self, idx):
        """Row idx's frame as an RGB float tensor in 0..1, before normalization."""
        bgr = cv2.imdecode(np.frombuffer(self.images[idx], np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"row {idx} has an undecodable image")
        return to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
