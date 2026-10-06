"""The part of a gripper camera model that is not about any one task, shared by visual_servoing
and basket.

    input   the gripper camera at its native 16:9, plus a small state vector

    trunk   frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only
              -> concat on channels                     (B, 3072, 18, 32)
            Conv2d(3072 -> 256, 1x1), GroupNorm, GELU   (B,  256, 18, 32)
            FiLM from the state vector
            self-attention blocks over 576 tokens       (B,  256, 18, 32)
            skip: concat the pre-attention map, 1x1 -> 256 (B,  256, 18, 32)

The cell grid spans CANVAS_SCALE times the frame, so a point just past an edge has a real
cell, and a position head reads it as a softmax over cells plus a log distance per cell:
(u, v, distance along the ray), which is a 3D point in the camera frame.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nf_robot.ml.dino_trunk import SharedTrunkMixin
from nf_robot.ml.grid_head import (
    AttentionBlock, FiLM, attend, cell_loss, gather_cells, local_centroid, masked_mean,
)

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-base"
# (width, height). 32x18 tokens at /14, and exactly the gripper camera's native 16:9.
DEFAULT_IMAGE_SIZE = (448, 252)
# The head predicts over 1.25x the frame extent so a target just past an edge has a real
# cell; must match mine_teleop.CANVAS_SCALE.
CANVAS_SCALE = 1.25
# Intrinsics of the 684x384 wide gripper stream, as fractions of the frame.
FOCAL_NORM = (284.84 / 684.0, 286.27 / 384.0)
PRINCIPAL_NORM = (342.0 / 684.0, 192.0 / 384.0)
# Width in cells (~18px) of the Gaussian the cell head trains against, matched to label
# precision; 0 is one-hot.
CELL_SIGMA = 1.0


def uv_to_cell(uv, grid):
    """Normalized frame coordinates (0..1 visible) to continuous canvas cell coordinates."""
    half = (CANVAS_SCALE - 1.0) / 2.0
    scale = uv.new_tensor([grid[1], grid[0]]) / CANVAS_SCALE
    return (uv + half) * scale


def cell_to_uv(cell, grid):
    """The inverse of uv_to_cell."""
    half = (CANVAS_SCALE - 1.0) / 2.0
    scale = cell.new_tensor([grid[1], grid[0]]) / CANVAS_SCALE
    return cell / scale - half


def camera_point(uv, distance):
    """(u, v, distance along the ray) as a 3D point in the camera's optical frame, in metres."""
    fx, fy = FOCAL_NORM
    cx, cy = PRINCIPAL_NORM
    x = (uv[..., 0] - cx) / fx
    y = (uv[..., 1] - cy) / fy
    ray = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return ray / ray.norm(dim=-1, keepdim=True) * distance.unsqueeze(-1)


class GripperGridNet(SharedTrunkMixin, nn.Module):
    """Frozen DINOv2 patch features -> a state-conditioned, attended map of cells, for a
    subclass to put its heads on."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 fuse_layers=4, width=256, attention_layers=3, heads=8, freeze=True,
                 state_dim=3):
        super().__init__()
        trunk = self._init_trunk(backbone_id, freeze)
        self.backbone_id = backbone_id
        self.image_size = tuple(image_size)
        self.fuse_layers = fuse_layers
        self.freeze = freeze
        self.state_dim = state_dim
        self.width = width

        config = trunk.config
        self.patch_size = config.patch_size
        width_px, height_px = self.image_size
        if width_px % self.patch_size or height_px % self.patch_size:
            raise ValueError(f"image_size {self.image_size} is not a multiple of patch {self.patch_size}")
        self.token_grid = (height_px // self.patch_size, width_px // self.patch_size)
        self.grid = self.token_grid
        self.hidden = config.hidden_size

        self.stem = nn.Sequential(
            nn.Conv2d(self.hidden * fuse_layers, width, 1), nn.GroupNorm(32, width), nn.GELU())
        self.film = FiLM(state_dim, width)
        # Learned position embedding for attention over the canvas.
        self.pos = nn.Parameter(torch.zeros(1, self.token_grid[0] * self.token_grid[1], width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attention = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(attention_layers)])
        # Skip connection so each spatial head sees the local pre-attention features beside
        # the attended ones.
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(width * 2, width, 1), nn.GroupNorm(32, width), nn.GELU())

    @property
    def global_dim(self):
        """Width of the global [CLS]/register vector features() returns."""
        return self.hidden * 2

    def features(self, pixel_values):
        """Fused patch features as a map, plus the global [CLS]/register vector."""
        rows, cols = self.token_grid
        tokens, last = self.patch_token_map(pixel_values, rows, cols)
        cls = last[:, 0]
        extras = last[:, 1:-rows * cols]
        registers = extras.mean(dim=1) if extras.shape[1] else torch.zeros_like(cls)
        return tokens, torch.cat([cls, registers], dim=-1)

    def cell_map(self, pixel_values, state):
        """The (B, width, rows, cols) map the spatial heads read, plus the global vector."""
        tokens, global_vec = self.features(pixel_values)
        local = self.film(self.stem(tokens), state)
        x = self.skip_fuse(torch.cat([local, attend(local, self.pos, self.attention)], dim=1))
        return x, global_vec


def position_losses(logits, log_distance, target_uv, target_range_m, has_uv, grid,
                    cell_sigma=CELL_SIGMA):
    """The position head's three losses - cell, centroid and log distance - each averaged
    over rows with a position label, plus the window weights and window of the true cell
    for heads that average over it."""
    rows, cols = grid
    cell = uv_to_cell(target_uv, grid)
    cx = cell[:, 0].floor().clamp(0, cols - 1).long()
    cy = cell[:, 1].floor().clamp(0, rows - 1).long()
    index = cy * cols + cx

    parts = {}
    # Only the cell head is softened.
    parts["cell"], _ = masked_mean(cell_loss(logits, cell, grid, index, cell_sigma), has_uv)

    # Train the windowed centre of mass directly, clamped to the outermost cell centres.
    centroid, window_weights, window = local_centroid(logits, index)
    reachable = cell.clamp(min=0.5).minimum(cell.new_tensor([cols - 0.5, rows - 0.5]))
    parts["centroid"], _ = masked_mean(
        F.smooth_l1_loss(centroid, reachable, reduction="none").mean(dim=1), has_uv)

    # Log metres, since range error is relative.
    predicted_log = gather_cells(log_distance.unsqueeze(1), index).squeeze(-1)
    target_log = target_range_m.clamp(min=1e-3).log()
    parts["distance"], _ = masked_mean(
        F.smooth_l1_loss(predicted_log, target_log, reduction="none"), has_uv)
    return parts, window_weights, window


def decode_position(logits, log_distance, grid, index):
    """(uv, distance, window weights, window) at one cell per batch item."""
    cell, weights, window = local_centroid(logits, index)
    distance = gather_cells(log_distance.unsqueeze(1), index).squeeze(-1).exp()
    return cell_to_uv(cell, grid), distance, weights, window
