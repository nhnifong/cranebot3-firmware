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
            skip: concat the pre-attention map, 1x1 -> 256 (B,  256, 18, 32)

    heads   1. target position, 3D, in the gripper camera frame: the centre of mass
               of the cell softmax in a window around the winning cell
            2. grasp axis, 2 channels, averaged over that same window
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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nf_robot.ml.dino_trunk import SharedTrunkMixin, drop_trunk_weights

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-base"
# (width, height). 32x18 tokens at /14, and exactly the gripper camera's native 16:9.
DEFAULT_IMAGE_SIZE = (448, 252)
# The head predicts over 1.25x the frame extent, so normalized coordinates run
# -0.125..1.125 and an object just past an edge has a real cell instead of being clamped
# to the border. That case is the whole reason this model exists. Must match
# mine_teleop.CANVAS_SCALE, which decides which labels the dataset keeps.
CANVAS_SCALE = 1.25
# Cells are the attention's own tokens: no upsampling, so 18x32 cells of 17.5px each.
# Precision below a cell comes from the centre of mass of the softmax in a window of
# this many cells either side of the winner, not from a separate offset head.
CENTROID_RADIUS = 2
# Shape the spatial close head reads the cell grid at: channels it reduces to, then the
# coarse grid it pools to. Pooled to a grid rather than to a vector because that is the
# whole point of moving the head here - "is there something between the fingers" is a
# question about one part of the frame, and a mean over the whole map cannot tell the
# bottom centre from anywhere else. 4x6 over 18x32 cells keeps the jaws in their own
# bins while staying small enough to flatten. On the 18x32 grid the bins are uneven
# (adaptive_avg_pool2d below), which is fine.
CLOSE_CHANNELS = 32
CLOSE_POOL = (4, 6)
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


def adaptive_avg_pool2d(x, size):
    """F.adaptive_avg_pool2d with the same bins, for any input size on any device.

    MPS refuses inputs that are not a multiple of the output, and 64 cells into 6 bins
    is not. Bins are rectangles of uniform weight, so pooling rows then columns is exact.
    """
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
        # Learned position embedding: after the 1x1 conv the backbone's own position
        # information is still there, but the attention reasons about the canvas rather
        # than the image and does better with its own.
        self.pos = nn.Parameter(torch.zeros(1, self.token_grid[0] * self.token_grid[1], width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attention = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(attention_layers)])

        channels = width

        # The skip connection: the map from before attention, concatenated with the map
        # after it and fused, so every spatial head reads "what is in this cell" beside
        # "what the whole image says about this cell". The attention blocks' own residual
        # stream does not provide this - by the last block it has been added to three
        # times, and a head reading it cannot tell the two apart. Without the local half a
        # cell's logit is whatever attention writes there, and pooling the positions of
        # several objects writes one peak at their mean.
        self.skip = skip
        if skip:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(width * 2, width, 1), nn.GroupNorm(32, width), nn.GELU())

        # Head 1: one softmax over canvas cells, plus log distance. Sub-cell position is
        # the softmax's local centre of mass; see local_centroid.
        self.logit_head = nn.Conv2d(channels, 1, 1)
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
        #
        # close_heads adds two more of the same kind, and they replace the finger head's
        # job rather than joining it. A rate says "how fast to move the fingers now",
        # which a teleoperator's thumb answers differently every frame; these two say
        # *when* to start and *how hard to end up squeezing*, which is what a grasp
        # actually consists of. The finger head stays in the output either way, so one
        # loader and one deployment path serve both kinds of checkpoint.
        self.close_heads = close_heads
        # Whether the close question is asked of the cell grid instead of the pooled
        # vector. It is a spatial test - something between the fingers, square on, near
        # enough - and the bottom centre of the frame is where "between the fingers" is.
        # A [CLS] vector has been averaged over the whole image before the head sees it,
        # so what survives is that the frame contains a graspable thing, not that this
        # one is in the jaws. Only meaningful alongside the close heads themselves.
        self.spatial_close = bool(spatial_close and close_heads)
        global_dim = hidden * 2 + state_dim
        self.global_outputs = (4 if self.spatial_close else 5) if close_heads else 3
        self.global_head = nn.Sequential(
            # LayerNorm first, and it is load-bearing: the trunk's [CLS] comes out with a
            # large norm, which drives the finger head's tanh straight into saturation
            # where its gradient is zero and it never trains at all. The spatial heads
            # do not have this problem because GroupNorm rescales the trunk for them.
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 256), nn.GELU(), nn.Linear(256, self.global_outputs))

        if self.spatial_close:
            # State is concatenated after the pool as well as being FiLMed into the map
            # upstream: the rangefinder is most of "close enough", and a direct path to
            # it costs one small matrix.
            self.close_reduce = nn.Conv2d(channels, CLOSE_CHANNELS, 1)
            close_dim = CLOSE_CHANNELS * CLOSE_POOL[0] * CLOSE_POOL[1] + state_dim
            self.close_head = nn.Sequential(
                nn.LayerNorm(close_dim),
                nn.Linear(close_dim, 256), nn.GELU(), nn.Linear(256, 1))

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
        local = self.film(self.stem(tokens), state)

        rows, cols = self.token_grid
        seq = local.flatten(2).transpose(1, 2) + self.pos
        for block in self.attention:
            seq = block(seq)
        x = seq.transpose(1, 2).reshape(local.shape[0], -1, rows, cols)
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
            # "Should the close have begun by now", as a logit, and the grip force the
            # operator ended up carrying this object with. The pressure is a softplus so
            # it cannot be predicted negative, which is not a force the gripper can hold.
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

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.trunk.eval()  # a frozen backbone must not update its norm statistics
        return self


def gather_cells(maps, index):
    """Pick one cell per batch item out of a (B, C, H, W) map, by flat cell index."""
    flat = maps.flatten(2)
    return flat.gather(2, index[:, None, None].expand(-1, flat.shape[1], -1)).squeeze(-1)


def local_centroid(logits, index, radius=CENTROID_RADIUS):
    """Centre of mass of the cell softmax in a window around one cell per batch item.

    Returns (cell, weights, window): the continuous cell coordinate (cell centres at
    i + 0.5, as uv_to_cell produces), the (B, K) softmax weights over the window, and the
    (B, K) flat indices of the window's cells for gathering other maps with the same
    weights. Cells off the grid get zero weight, so a winner on the border is pulled
    inward only by what is really there.

    A softmax over just the window's logits is the global softmax renormalized to the
    window, so this is "the probability mass near the peak, averaged" - and restricting
    it to a window is what keeps a second object elsewhere from dragging the answer onto
    the floor between them.
    """
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
    """Head outputs -> (uv, distance, axis angle, score), all batched, top_k per item.

    Peaks rather than the expectation over the map: averaging two candidate objects
    would land the prediction on the empty floor between them - the same reason
    ortho_target decodes the way it does. The expectation is taken only locally, in a
    window around each peak, which is where the sub-cell position comes from.
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
        cell, weights, window = local_centroid(logits, idx)
        uv.append(cell_to_uv(cell, grid))
        distance.append(gather_cells(outputs["log_distance"].unsqueeze(1), idx).squeeze(-1).exp())
        # The raw (sin 2t, cos 2t) vectors, not their angles, are averaged: neighbours
        # that disagree cancel, which shortens the result and so lowers the concentration
        # reported below - disagreement reads as uncertainty, as it should.
        axis = window_average(outputs["axis"], weights, window)
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
    # These default off while training defaults them on, and the asymmetry is the point:
    # here the question is what a checkpoint that does not mention them was built with,
    # and the answer is the shape the key's absence describes.
    # Absent in every checkpoint written before these heads existed, which is what makes
    # those checkpoints keep loading: no key, no extra heads, same three outputs.
    model = VisualServoNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        fuse_layers=checkpoint["fuse_layers"], attention_layers=checkpoint["attention_layers"],
        freeze=freeze, close_heads=checkpoint.get("close_heads", False),
        spatial_close=checkpoint.get("spatial_close", False),
        # absent in checkpoints trained before the skip connection existed
        skip=checkpoint.get("skip", False),
    ).to(device)
    state = checkpoint["state_dict"]
    if freeze:
        # Checkpoints written before the trunk was shared still carry it; verify they
        # really are the pretrained weights when the checkpoint does not say so itself.
        state = drop_trunk_weights(state, model.trunk, verify="freeze" not in checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint
