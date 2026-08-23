#!/usr/bin/env python

"""The visual servoing network: frozen DINOv2 patch tokens -> where the object is.

The model predicts where the target is, not how fast to move, because the position
Something downstream turns a position into a velocity, the way
observer.py's _center_card_in_view already does.

    input   448x252 RGB, the gripper camera at its native aspect
            plus laser_rangefinder, finger_angle and target_force

    trunk   frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only
              -> concat on channels                     (B, 3072, 18, 32)
            Conv2d(3072 -> 256, 1x1), GroupNorm, GELU   (B,  256, 18, 32)
            FiLM from the state vector
            self-attention blocks over 576 tokens       (B,  256, 18, 32)
            bilinear x2, Conv 3x3 -> 128, GN, GELU      (B,  128, 36, 64)

    heads   1. target position, 3D, in the gripper camera frame
            2. grasp axis, 2 channels, read from the winning cell
            3. finger speed, scalar in [-1, 1], from the global vector
            4. probability any graspable target is present
            5. probability we are currently holding something

The self-attention blocks are the one real addition over OrthoTargetNet in
ortho_target.py, whose decoder this otherwise follows.

The input size is tied to the backbone's patch size and to the size the dataset was
written at, because nothing here resizes a stored frame. 448x252 is 14 x (32, 18).
Swapping backbones therefore means re-mining; readme.md's DINOv3 footnote has the
details and the failure mode.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nf_robot.ml.dino_trunk import SharedTrunkMixin, drop_trunk_weights

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-base"
# (width, height). 32x18 tokens at /14, and exactly the gripper camera's native 16:9.
DEFAULT_IMAGE_SIZE = (448, 252)
# The head predicts over 1.5x the frame extent, so normalized coordinates run
# -0.25..1.25 and an object just past an edge has a real cell instead of being clamped
# to the border. That case is the whole reason this model exists.
CANVAS_SCALE = 1.5
# Cells per token, i.e. how far the decoder upsamples. 2 gives 36x64 cells of about
# 10.5px each, and the offset head takes precision below that.
CELLS_PER_TOKEN = 2
# laser_rangefinder, finger_angle, target_force. Deliberately not the previous velocity
# (the shortcut that teaches "keep doing what you were doing") and not the measured
# finger pressure (the answer head 5 is supposed to work out from the image).
STATE_DIM = 3
# Intrinsics of the 684x384 wide gripper stream, as fractions of the frame, so they
# apply at any resolution the frame is stored or resized to.
FOCAL_NORM = (439.31834658631243 / 684.0, 461.5621083718772 / 384.0)
PRINCIPAL_NORM = (342.0 / 684.0, 192.0 / 384.0)


def uv_to_cell(uv, grid):
    """Normalized frame coordinates to continuous canvas cell coordinates.

    uv is (..., 2) with 0..1 spanning the visible frame; the canvas spans
    -0.25..1.25, so a target off the bottom edge still lands on a real cell.
    """
    half = (CANVAS_SCALE - 1.0) / 2.0
    scale = uv.new_tensor([grid[1], grid[0]]) / CANVAS_SCALE
    return (uv + half) * scale


def cell_to_uv(cell, grid):
    """The inverse of uv_to_cell."""
    half = (CANVAS_SCALE - 1.0) / 2.0
    scale = cell.new_tensor([grid[1], grid[0]]) / CANVAS_SCALE
    return cell / scale - half


def camera_point(uv, distance):
    """(u, v, distance) as a 3D point in the camera's optical frame, in metres.

    The head's three numbers are two angles and a range; this is the pinhole model that
    turns them into the position the robot actually wants. Distance is along the ray,
    not depth along the optical axis, so the ray is normalized before scaling.
    """
    fx, fy = FOCAL_NORM
    cx, cy = PRINCIPAL_NORM
    x = (uv[..., 0] - cx) / fx
    y = (uv[..., 1] - cy) / fy
    ray = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return ray / ray.norm(dim=-1, keepdim=True) * distance.unsqueeze(-1)


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


class AttentionBlock(nn.Module):
    """Pre-norm self-attention plus MLP over the flattened token grid."""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisualServoNet(SharedTrunkMixin, nn.Module):
    """Frozen DINOv2 patch features -> target position, grasp axis, finger, flags."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 fuse_layers=4, width=256, attention_layers=3, heads=8, freeze=True,
                 state_dim=STATE_DIM):
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
        self.grid = (self.token_grid[0] * CELLS_PER_TOKEN, self.token_grid[1] * CELLS_PER_TOKEN)

        hidden = config.hidden_size
        self.stem = nn.Sequential(
            nn.Conv2d(hidden * fuse_layers, width, 1), nn.GroupNorm(32, width), nn.GELU())
        self.film = FiLM(state_dim, width)
        # Learned position embedding: after the 1x1 conv the backbone's own position
        # information is still there, but the attention reasons about the canvas rather
        # than the image and does better with its own.
        self.pos = nn.Parameter(torch.zeros(1, self.token_grid[0] * self.token_grid[1], width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attention = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(attention_layers)])

        decoder, channels = [], width
        for _ in range(int(math.log2(CELLS_PER_TOKEN))):
            nxt = max(64, channels // 2)
            decoder += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, nxt, 3, padding=1), nn.GroupNorm(32, nxt), nn.GELU(),
            ]
            channels = nxt
        self.decoder = nn.Sequential(*decoder)

        # Head 1: one softmax over canvas cells, plus sub-cell offset and log distance.
        self.logit_head = nn.Conv2d(channels, 1, 1)
        self.offset_head = nn.Conv2d(channels, 2, 1)
        # log metres, per cell: with two objects at different heights in frame there is
        # no single correct distance for the image
        self.distance_head = nn.Conv2d(channels, 1, 1)
        # Head 2: (sin 2t, cos 2t). Doubled because a two-finger grasp axis is
        # pi-periodic, and regressing the angle itself puts a wraparound discontinuity
        # in the middle of the label space.
        self.axis_head = nn.Conv2d(channels, 2, 1)

        # Heads 3-5 read the whole image, not a location. Whether to close the fingers
        # is a property of the frame: by the time it matters the object usually fills or
        # blinds the camera, and we are aiming at a chunky lump of towel rather than a
        # centred object.
        global_dim = hidden * 2 + state_dim
        self.global_head = nn.Sequential(
            # LayerNorm first, and it is load-bearing: the trunk's [CLS] comes out with a
            # large norm, which drives the finger head's tanh straight into saturation
            # where its gradient is zero and it never trains at all. The spatial heads
            # do not have this problem because GroupNorm rescales the trunk for them.
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 256), nn.GELU(), nn.Linear(256, 3))

    def features(self, pixel_values):
        """Fused patch features as a map, plus the global [CLS]/register vector."""
        with torch.set_grad_enabled(self.training and not self.freeze):
            out = self.trunk(pixel_values, output_hidden_states=True)
        rows, cols = self.token_grid
        n_patches = rows * cols
        # [CLS] and the register tokens lead the sequence; patches are always the tail.
        feats = [h[:, -n_patches:, :] for h in out.hidden_states[-self.fuse_layers:]]
        x = torch.cat(feats, dim=-1).transpose(1, 2)
        last = out.hidden_states[-1]
        cls = last[:, 0]
        extras = last[:, 1:-n_patches]
        registers = extras.mean(dim=1) if extras.shape[1] else torch.zeros_like(cls)
        return x.reshape(x.shape[0], x.shape[1], rows, cols), torch.cat([cls, registers], dim=-1)

    def forward(self, pixel_values, state):
        """Returns a dict of raw head outputs; see decode() for what they mean."""
        tokens, global_vec = self.features(pixel_values)
        x = self.film(self.stem(tokens), state)

        rows, cols = self.token_grid
        seq = x.flatten(2).transpose(1, 2) + self.pos
        for block in self.attention:
            seq = block(seq)
        x = seq.transpose(1, 2).reshape(x.shape[0], -1, rows, cols)

        x = self.decoder(x)
        flags = self.global_head(torch.cat([global_vec, state], dim=-1))
        return {
            "logits": self.logit_head(x).squeeze(1),
            "offsets": self.offset_head(x),
            "log_distance": self.distance_head(x).squeeze(1),
            "axis": self.axis_head(x),
            "finger": torch.tanh(flags[:, 0]),
            "present_logit": flags[:, 1],
            "holding_logit": flags[:, 2],
        }

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.trunk.eval()  # a frozen backbone must not update its norm statistics
        return self


def gather_cells(maps, index):
    """Pick one cell per batch item out of a (B, C, H, W) map, by flat cell index."""
    flat = maps.flatten(2)
    return flat.gather(2, index[:, None, None].expand(-1, flat.shape[1], -1)).squeeze(-1)


def decode(outputs, grid, top_k=1, nms_radius=2):
    """Head outputs -> (uv, distance, axis angle, score), all batched, top_k per item.

    Peaks rather than the expectation over the map: averaging two candidate objects
    would land the prediction on the empty floor between them - the same reason
    ortho_target decodes the way it does.
    """
    logits = outputs["logits"]
    batch, rows, cols = logits.shape
    prob = logits.flatten(1).softmax(1).view(batch, 1, rows, cols)

    if top_k > 1:
        # Suppress everything that is not a local maximum so the k results are k
        # distinct candidates rather than k cells of one blob.
        pooled = F.max_pool2d(prob, nms_radius * 2 + 1, stride=1, padding=nms_radius)
        prob = torch.where(prob >= pooled, prob, torch.zeros_like(prob))

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    uv, distance, angle, concentration = [], [], [], []
    for k in range(top_k):
        idx = index[:, k]
        cx = (idx % cols).float()
        cy = torch.div(idx, cols, rounding_mode="floor").float()
        offset = gather_cells(outputs["offsets"], idx).sigmoid()
        cell = torch.stack([cx, cy], dim=-1) + offset
        uv.append(cell_to_uv(cell, grid))
        distance.append(gather_cells(outputs["log_distance"].unsqueeze(1), idx).squeeze(-1).exp())
        axis = gather_cells(outputs["axis"], idx)
        angle.append(torch.atan2(axis[:, 0], axis[:, 1]) / 2.0)
        # The length of the axis vector, which the von Mises objective in train.py trains
        # as the head's concentration: how sure it is, not just what it thinks. atan2
        # throws it away, so it is returned beside the angle rather than recovered later.
        # Near zero means "no opinion", and a consumer turning a wrist should want one.
        concentration.append(axis.norm(dim=1))
    return (torch.stack(uv, dim=1), torch.stack(distance, dim=1),
            torch.stack(angle, dim=1), scores, torch.stack(concentration, dim=1))


@torch.no_grad()
def predict(model, images, state, top_k=1):
    """Everything the robot wants from one frame, decoded and unbatched-friendly."""
    model.eval()
    outputs = model(images, state)
    uv, distance, angle, scores, concentration = decode(outputs, model.grid, top_k=top_k)
    return {
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


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    freeze = checkpoint.get("freeze", True)
    model = VisualServoNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        fuse_layers=checkpoint["fuse_layers"], attention_layers=checkpoint["attention_layers"],
        freeze=freeze,
    ).to(device)
    state = checkpoint["state_dict"]
    if freeze:
        # Checkpoints written before the trunk was shared still carry it; verify they
        # really are the pretrained weights when the checkpoint does not say so itself.
        state = drop_trunk_weights(state, model.trunk, verify="freeze" not in checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint
