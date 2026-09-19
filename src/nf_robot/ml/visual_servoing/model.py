#!/usr/bin/env python

"""The visual servoing network: frozen DINOv2 patch tokens -> where the object is.

    input   448x252 RGB, the gripper camera at its native aspect
            plus laser_rangefinder, finger_angle and target_force

    trunk   frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only
              -> concat on channels                     (B, 3072, 18, 32)
            Conv2d(3072 -> 256, 1x1), GroupNorm, GELU   (B,  256, 18, 32)
            FiLM from the state vector
            self-attention blocks over 576 tokens       (B,  256, 18, 32)
            skip: concat the pre-attention map, 1x1 -> 256 (B,  256, 18, 32)

    heads   1. target position, 3D, in the gripper camera frame: the centre of mass
               of the cell softmax in a window around the winning cell
            2. grasp axis, 2 channels, averaged over that same window
            3. finger speed, scalar in [-1, 1], from the global vector
            4. probability any graspable target is present
            5. probability we are currently holding something
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nf_robot.ml.dino_trunk import SharedTrunkMixin, load_head_state
from nf_robot.ml.grid_head import AttentionBlock, attend, local_maxima

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-base"
# (width, height). 32x18 tokens at /14, and exactly the gripper camera's native 16:9.
DEFAULT_IMAGE_SIZE = (448, 252)
# The head predicts over 1.25x the frame extent so a target just past an edge has a real
# cell; must match mine_teleop.CANVAS_SCALE.
CANVAS_SCALE = 1.25
# Cells either side of the winner whose softmax centre of mass gives the sub-cell position.
CENTROID_RADIUS = 2
# Channels and pooled grid of the spatial close head, coarse enough to flatten but keeping
# the jaws in their own bins.
CLOSE_CHANNELS = 32
CLOSE_POOL = (4, 6)
# laser_rangefinder, finger_angle, target_force; deliberately not velocity or finger
# pressure.
STATE_DIM = 3
# Intrinsics of the 684x384 wide gripper stream, as fractions of the frame.
FOCAL_NORM = (439.31834658631243 / 684.0, 461.5621083718772 / 384.0)
PRINCIPAL_NORM = (342.0 / 684.0, 192.0 / 384.0)


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


def adaptive_avg_pool2d(x, size):
    """F.adaptive_avg_pool2d that also works on MPS for sizes that don't divide evenly."""
    for dim, out in ((-2, size[0]), (-1, size[1])):
        n = x.shape[dim]
        x = torch.stack([x.narrow(dim, i * n // out, -(-(i + 1) * n // out) - i * n // out).mean(dim)
                         for i in range(out)], dim=dim)
    return x


class FiLM(nn.Module):
    """Per-channel scale and shift on a feature map, conditioned on the state vector."""

    def __init__(self, state_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.GELU(), nn.Linear(128, channels * 2))
        # start as the identity so an untrained FiLM does not scramble the features
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, state):
        scale, shift = self.net(state).chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class VisualServoNet(SharedTrunkMixin, nn.Module):
    """Frozen DINOv2 patch features -> target position, grasp axis, finger, flags."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 fuse_layers=4, width=256, attention_layers=3, heads=8, freeze=True,
                 state_dim=STATE_DIM, close_heads=False, spatial_close=False, skip=True):
        super().__init__()
        trunk = self._init_trunk(backbone_id, freeze)
        self.backbone_id = backbone_id
        self.image_size = tuple(image_size)
        self.fuse_layers = fuse_layers
        self.freeze = freeze
        self.state_dim = state_dim

        config = trunk.config
        self.patch_size = config.patch_size
        width_px, height_px = self.image_size
        if width_px % self.patch_size or height_px % self.patch_size:
            raise ValueError(f"image_size {self.image_size} is not a multiple of patch {self.patch_size}")
        self.token_grid = (height_px // self.patch_size, width_px // self.patch_size)
        self.grid = self.token_grid

        hidden = config.hidden_size
        self.stem = nn.Sequential(
            nn.Conv2d(hidden * fuse_layers, width, 1), nn.GroupNorm(32, width), nn.GELU())
        self.film = FiLM(state_dim, width)
        # Learned position embedding for attention over the canvas.
        self.pos = nn.Parameter(torch.zeros(1, self.token_grid[0] * self.token_grid[1], width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attention = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(attention_layers)])

        channels = width

        # Skip connection so each spatial head sees the local pre-attention features beside
        # the attended ones.
        self.skip = skip
        if skip:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(width * 2, width, 1), nn.GroupNorm(32, width), nn.GELU())

        # Head 1: one softmax over canvas cells, plus log distance.
        self.logit_head = nn.Conv2d(channels, 1, 1)
        # Log metres per cell, since objects at different heights have different distances.
        self.distance_head = nn.Conv2d(channels, 1, 1)
        # Head 2: (sin 2t, cos 2t), because the grasp axis is pi-periodic.
        self.axis_head = nn.Conv2d(channels, 2, 1)

        # Heads 3-5 read the whole image; close_heads adds when-to-close and how-hard heads
        # beside the finger rate.
        self.close_heads = close_heads
        # Ask the close question of the cell grid, where the jaws are, rather than the
        # pooled vector.
        self.spatial_close = bool(spatial_close and close_heads)
        global_dim = hidden * 2 + state_dim
        self.global_outputs = (4 if self.spatial_close else 5) if close_heads else 3
        self.global_head = nn.Sequential(
            # LayerNorm first, or the large [CLS] norm saturates the finger head's tanh.
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 256), nn.GELU(), nn.Linear(256, self.global_outputs))

        if self.spatial_close:
            # State also goes straight in after the pool, since the rangefinder is most of
            # "close enough".
            self.close_reduce = nn.Conv2d(channels, CLOSE_CHANNELS, 1)
            close_dim = CLOSE_CHANNELS * CLOSE_POOL[0] * CLOSE_POOL[1] + state_dim
            self.close_head = nn.Sequential(
                nn.LayerNorm(close_dim),
                nn.Linear(close_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def features(self, pixel_values):
        """Fused patch features as a map, plus the global [CLS]/register vector."""
        rows, cols = self.token_grid
        tokens, last = self.patch_token_map(pixel_values, rows, cols)
        cls = last[:, 0]
        extras = last[:, 1:-rows * cols]
        registers = extras.mean(dim=1) if extras.shape[1] else torch.zeros_like(cls)
        return tokens, torch.cat([cls, registers], dim=-1)

    def forward(self, pixel_values, state):
        """Returns a dict of raw head outputs; see decode() for what they mean."""
        tokens, global_vec = self.features(pixel_values)
        local = self.film(self.stem(tokens), state)

        x = attend(local, self.pos, self.attention)
        if self.skip:
            x = self.skip_fuse(torch.cat([local, x], dim=1))

        flags = self.global_head(torch.cat([global_vec, state], dim=-1))
        out = {
            "logits": self.logit_head(x).squeeze(1),
            "log_distance": self.distance_head(x).squeeze(1),
            "axis": self.axis_head(x),
            "finger": torch.tanh(flags[:, 0]),
            "present_logit": flags[:, 1],
            "holding_logit": flags[:, 2],
        }
        if self.close_heads:
            # Close onset as a logit, and grip pressure as a softplus so it can't go
            # negative.
            if self.spatial_close:
                cells = F.gelu(self.close_reduce(x))
                pooled = adaptive_avg_pool2d(cells, CLOSE_POOL).flatten(1)
                out["close_logit"] = self.close_head(
                    torch.cat([pooled, state], dim=-1)).squeeze(-1)
                out["grasp_pressure"] = F.softplus(flags[:, 3])
            else:
                out["close_logit"] = flags[:, 3]
                out["grasp_pressure"] = F.softplus(flags[:, 4])
        return out


def gather_cells(maps, index):
    """Pick one cell per batch item out of a (B, C, H, W) map, by flat cell index."""
    flat = maps.flatten(2)
    return flat.gather(2, index[:, None, None].expand(-1, flat.shape[1], -1)).squeeze(-1)


def local_centroid(logits, index, radius=CENTROID_RADIUS):
    """Centre of mass of the cell softmax in a window around one cell per batch item, as
    (cell, weights, window)."""
    batch, rows, cols = logits.shape
    steps = torch.arange(-radius, radius + 1, device=logits.device)
    dy, dx = torch.meshgrid(steps, steps, indexing="ij")
    dy, dx = dy.flatten(), dx.flatten()
    cy = torch.div(index, cols, rounding_mode="floor")[:, None] + dy
    cx = (index % cols)[:, None] + dx
    valid = (cy >= 0) & (cy < rows) & (cx >= 0) & (cx < cols)
    window = cy.clamp(0, rows - 1) * cols + cx.clamp(0, cols - 1)
    local = logits.flatten(1).gather(1, window).masked_fill(~valid, float("-inf"))
    weights = local.softmax(dim=1)
    centres = torch.stack([cx, cy], dim=-1).to(logits.dtype) + 0.5
    return (weights.unsqueeze(-1) * centres).sum(dim=1), weights, window


def window_average(maps, weights, window):
    """Average a (B, C, H, W) map over local_centroid's window with its weights."""
    flat = maps.flatten(2)
    gathered = flat.gather(2, window[:, None, :].expand(-1, flat.shape[1], -1))
    return (gathered * weights[:, None, :]).sum(dim=2)


def decode(outputs, grid, top_k=1, nms_radius=CENTROID_RADIUS):
    """Head outputs -> (uv, distance, axis angle, score, concentration), top_k peaks per item."""
    logits = outputs["logits"]
    batch, rows, cols = logits.shape
    prob = logits.flatten(1).softmax(1).view(batch, 1, rows, cols)

    if top_k > 1:
        prob = local_maxima(prob, nms_radius)

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    uv, distance, angle, concentration = [], [], [], []
    for k in range(top_k):
        idx = index[:, k]
        cell, weights, window = local_centroid(logits, idx)
        uv.append(cell_to_uv(cell, grid))
        distance.append(gather_cells(outputs["log_distance"].unsqueeze(1), idx).squeeze(-1).exp())
        # Average the (sin 2t, cos 2t) vectors so disagreement shortens them.
        axis = window_average(outputs["axis"], weights, window)
        angle.append(torch.atan2(axis[:, 0], axis[:, 1]) / 2.0)
        # The axis vector's length is the head's concentration, near zero meaning no
        # opinion.
        concentration.append(axis.norm(dim=1))
    return (torch.stack(uv, dim=1), torch.stack(distance, dim=1),
            torch.stack(angle, dim=1), scores, torch.stack(concentration, dim=1))


@torch.no_grad()
def predict(model, images, state, top_k=1):
    """Everything the robot wants from one frame, decoded and unbatched-friendly."""
    model.eval()
    outputs = model(images, state)
    uv, distance, angle, scores, concentration = decode(outputs, model.grid, top_k=top_k)
    result = {
        "uv": uv,
        "distance_m": distance,
        "point_m": camera_point(uv, distance),
        "grasp_axis_rad": angle,
        "axis_concentration": concentration,
        "score": scores,
        "finger": outputs["finger"],
        "present": outputs["present_logit"].sigmoid(),
        "holding": outputs["holding_logit"].sigmoid(),
    }
    if "close_logit" in outputs:
        result["close"] = outputs["close_logit"].sigmoid()
        result["grasp_pressure"] = outputs["grasp_pressure"]
    return result


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    freeze = checkpoint.get("freeze", True)
    # Heads a checkpoint doesn't mention were not built, so these default off here.
    model = VisualServoNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        fuse_layers=checkpoint["fuse_layers"], attention_layers=checkpoint["attention_layers"],
        freeze=freeze, close_heads=checkpoint.get("close_heads", False),
        spatial_close=checkpoint.get("spatial_close", False),
        # absent in checkpoints trained before the skip connection existed
        skip=checkpoint.get("skip", False),
    ).to(device)
    load_head_state(model, checkpoint)
    model.eval()
    return model, checkpoint
