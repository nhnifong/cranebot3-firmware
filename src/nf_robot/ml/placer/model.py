#!/usr/bin/env python

"""Where to drop this item: frozen DINOv2 tokens of the item and of the room -> one cell.

    input   item      448x252 RGB, the gripper camera just before the grasp
            overhead  448x448 RGB, the ortho floor view at that moment

    trunk   frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only, both images
              -> overhead map                              (B, 3072, 32, 32)
              -> item map                                  (B, 3072, 18, 32)
            Conv2d(3072 -> 256, 1x1), GroupNorm, GELU on each

    merge   FiLM the overhead map from the item's [CLS]/register vector, and carry the
            item as 12 pooled context tokens through the attention (--merge picks either)

    heads   self-attention blocks over the 1024 overhead tokens plus the context tokens
            skip: concat the pre-attention map, 1x1 -> 256
            one logit per overhead cell

The drop point is the centre of mass of the cell softmax in a window around the winning
cell, as in VisualServoNet - the cells are 15.6cm of floor, and that is what puts the
answer inside one.
"""

import torch
import torch.nn as nn

from nf_robot.ml.dino_trunk import SharedTrunkMixin, load_head_state
from nf_robot.ml.grid_head import AttentionBlock, local_maxima
from nf_robot.ml.ortho_target.model import ORTHO_EXTENT_M
from nf_robot.ml.visual_servoing.model import DEFAULT_BACKBONE, FiLM, adaptive_avg_pool2d, local_centroid

# Where a trained checkpoint is published, and where --local_models looks for it instead.
DROP_POINT_MODEL_REPOID = "naavox/drop_point"
DROP_POINT_MODEL_FILENAME = "drop_point.pth"
LOCAL_MODEL_PATH = "models/drop_point.pth"

# (width, height), both at the size the miner stores them.
OVERHEAD_SIZE = (448, 448)
ITEM_SIZE = (448, 252)
# Item context tokens, as (rows, cols) pooled off its patch grid.
ITEM_TOKENS = (3, 4)
MERGE_MODES = ("both", "film", "tokens")
CENTROID_RADIUS = 2


def cell_to_uv(cell, grid):
    """Continuous cell coordinates to normalized overhead coordinates."""
    return cell / cell.new_tensor([grid[1], grid[0]])


def uv_to_cell(uv, grid):
    """The inverse of cell_to_uv."""
    return uv * uv.new_tensor([grid[1], grid[0]])


def uv_to_metres(uv_error):
    """A normalized overhead distance as metres of floor."""
    return uv_error * ORTHO_EXTENT_M


class DropPointNet(SharedTrunkMixin, nn.Module):
    """Item snapshot + overhead view -> a softmax over overhead cells."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, overhead_size=OVERHEAD_SIZE,
                 item_size=ITEM_SIZE, fuse_layers=4, width=256, attention_layers=3, heads=8,
                 freeze=True, merge="both", item_tokens=ITEM_TOKENS):
        super().__init__()
        if merge not in MERGE_MODES:
            raise ValueError(f"unknown merge {merge!r}; expected one of {MERGE_MODES}")
        trunk = self._init_trunk(backbone_id, freeze)
        self.backbone_id = backbone_id
        self.overhead_size = tuple(overhead_size)
        self.item_size = tuple(item_size)
        self.fuse_layers = fuse_layers
        self.freeze = freeze
        self.merge = merge
        self.item_tokens = tuple(item_tokens)

        config = trunk.config
        self.patch_size = config.patch_size
        self.grid = self._token_grid(self.overhead_size)
        self.item_grid = self._token_grid(self.item_size)

        hidden = config.hidden_size
        def stem():
            return nn.Sequential(
                nn.Conv2d(hidden * fuse_layers, width, 1), nn.GroupNorm(32, width), nn.GELU())
        self.stem = stem()
        self.item_stem = stem()
        # The item as one vector, which is what says which container this belongs in.
        self.film = FiLM(hidden * 2, width)

        self.pos = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        # Tells the attention which tokens are the item rather than the room.
        self.context_pos = nn.Parameter(torch.zeros(1, item_tokens[0] * item_tokens[1], width))
        nn.init.trunc_normal_(self.context_pos, std=0.02)
        self.attention = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(attention_layers)])
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(width * 2, width, 1), nn.GroupNorm(32, width), nn.GELU())
        self.logit_head = nn.Conv2d(width, 1, 1)

    def _token_grid(self, size):
        width_px, height_px = size
        if width_px % self.patch_size or height_px % self.patch_size:
            raise ValueError(f"image_size {size} is not a multiple of patch {self.patch_size}")
        return (height_px // self.patch_size, width_px // self.patch_size)

    def features(self, pixel_values, grid):
        """Fused patch features as a map, plus the global [CLS]/register vector."""
        rows, cols = grid
        tokens, last = self.patch_token_map(pixel_values, rows, cols)
        cls = last[:, 0]
        extras = last[:, 1:-rows * cols]
        registers = extras.mean(dim=1) if extras.shape[1] else torch.zeros_like(cls)
        return tokens, torch.cat([cls, registers], dim=-1)

    def forward(self, item, overhead):
        """Returns a dict with one logit per overhead cell."""
        room_tokens, _ = self.features(overhead, self.grid)
        item_map, item_vector = self.features(item, self.item_grid)

        local = self.stem(room_tokens)
        if self.merge in ("both", "film"):
            local = self.film(local, item_vector)

        rows, cols = self.grid
        seq = local.flatten(2).transpose(1, 2) + self.pos
        if self.merge in ("both", "tokens"):
            context = adaptive_avg_pool2d(self.item_stem(item_map), self.item_tokens)
            seq = torch.cat([seq, context.flatten(2).transpose(1, 2) + self.context_pos], dim=1)
        for block in self.attention:
            seq = block(seq)
        attended = seq[:, :rows * cols].transpose(1, 2).reshape(local.shape[0], -1, rows, cols)

        x = self.skip_fuse(torch.cat([local, attended], dim=1))
        return {"logits": self.logit_head(x).squeeze(1)}


def decode(outputs, grid, top_k=1, nms_radius=CENTROID_RADIUS):
    """Head outputs -> (uv, score), top_k peaks per item, sub-cell by local centroid."""
    logits = outputs["logits"]
    batch, rows, cols = logits.shape
    prob = logits.flatten(1).softmax(1).view(batch, 1, rows, cols)
    if top_k > 1:
        prob = local_maxima(prob, nms_radius)

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    uv = []
    for k in range(top_k):
        cell, _, _ = local_centroid(logits, index[:, k], radius=nms_radius)
        uv.append(cell_to_uv(cell, grid))
    return torch.stack(uv, dim=1), scores


@torch.no_grad()
def predict(model, item, overhead, top_k=1):
    """Drop points for a batch, as normalized overhead coordinates and their scores."""
    model.eval()
    uv, scores = decode(model(item, overhead), model.grid, top_k=top_k)
    return {"uv": uv, "score": scores}


def load_model(device, local_models=False, revision=None):
    """The trained checkpoint on `device`, from models/ or from the hub at `revision`.

    `revision` is the hub commit to pin to; None takes the tip of main, which is what a
    model that has not been pinned yet gets.
    """
    if local_models:
        path = LOCAL_MODEL_PATH
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=DROP_POINT_MODEL_REPOID,
                               filename=DROP_POINT_MODEL_FILENAME, revision=revision)
    return load_checkpoint(path, device)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DropPointNet(
        backbone_id=checkpoint["backbone_id"], overhead_size=checkpoint["overhead_size"],
        item_size=checkpoint["item_size"], fuse_layers=checkpoint["fuse_layers"],
        attention_layers=checkpoint["attention_layers"], merge=checkpoint["merge"],
        freeze=checkpoint.get("freeze", True),
    ).to(device)
    load_head_state(model, checkpoint)
    model.eval()
    return model, checkpoint
