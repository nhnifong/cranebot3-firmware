#!/usr/bin/env python

"""The ortho targeting network: frozen DINOv2 patch tokens -> objectness per floor cell.

    input   448x448 RGB, the ortho floor view: 5m of floor per side, room origin centred

    trunk   frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only
              -> concat on channels                       (B, 3072,  32,  32)
            Conv2d(3072 -> 256, 1x1), GroupNorm, GELU     (B,  256,  32,  32)
            self-attention blocks over 1024 tokens        (B,  256,  32,  32)
            skip: concat the pre-attention map, 1x1 -> 256 (B,  256,  32,  32)
            2x (bilinear x2, 3x3 conv, GroupNorm, GELU)   (B,   64, 128, 128)

    heads   1. objectness per cell, each its own sigmoid
            2. sub-cell offset of a target within its cell, 2 channels
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nf_robot.ml.dino_trunk import SharedTrunkMixin, load_head_state
from nf_robot.ml.grid_head import AttentionBlock, attend, local_maxima
from nf_robot.ml.image_input import input_batch

# Metres of floor the ortho map spans per side, matching host/floor_view.py's EXTENT_M.
ORTHO_EXTENT_M = 5.0


def room_to_ortho_px(x_m, y_m, width, height, extent_m=ORTHO_EXTENT_M):
    """Room-frame metres -> ortho pixel coordinates, mirroring
    floor_view.generate_orthographic_floor_maps."""
    u = x_m * (width / extent_m) + width / 2.0
    v = -y_m * (height / extent_m) + height / 2.0
    return u, v


def ortho_px_to_room(u, v, width, height, extent_m=ORTHO_EXTENT_M):
    x_m = (u - width / 2.0) * extent_m / width
    y_m = -(v - height / 2.0) * extent_m / height
    return x_m, y_m


# ==========================================
# MODEL
# ==========================================

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-base"
# 448 = 14 x 32, so a /14 trunk gives a 32x32 token grid; a /16 trunk wants 512.
DEFAULT_IMAGE_SIZE = 448
DEFAULT_GRID = 128
# Self-attention blocks between the stem and the upsampling; old checkpoints without a count
# load as 0.
DEFAULT_ATTENTION_LAYERS = 3
# Concatenate the pre-attention map with the attended one; old checkpoints without the flag
# load without it.
DEFAULT_ATTENTION_SKIP = True
DEFAULT_MODEL_PATH = "models/ortho_target.pth"
# Width in cells of the Gaussian the cell head trains against (~6cm of floor).
CELL_SIGMA = 1.5
# Fallback objectness threshold for checkpoints that don't carry their own.
TARGET_THRESHOLD = 0.5


class OrthoTargetNet(SharedTrunkMixin, nn.Module):
    """Frozen DINOv2 patch features -> an independent sigmoid objectness per cell, plus a
    sub-cell offset."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 grid=DEFAULT_GRID, fuse_layers=4, width=256, freeze=True,
                 attention_layers=DEFAULT_ATTENTION_LAYERS, heads=8,
                 attention_skip=DEFAULT_ATTENTION_SKIP):
        super().__init__()
        trunk = self._init_trunk(backbone_id, freeze)
        self.backbone_id = backbone_id
        self.image_size = image_size
        self.grid = grid
        self.fuse_layers = fuse_layers
        self.freeze = freeze
        self.attention_layers = attention_layers
        self.attention_skip = bool(attention_skip and attention_layers)

        config = trunk.config
        self.patch_size = config.patch_size
        self.token_grid = image_size // self.patch_size
        if image_size % self.patch_size:
            raise ValueError(f"image_size {image_size} is not a multiple of patch {self.patch_size}")

        in_ch = config.hidden_size * fuse_layers
        # Two bilinear x2 steps take the 16px token grid to 4px cells at 512 input.
        ups = int(math.log2(grid / self.token_grid))
        if 2 ** ups * self.token_grid != grid:
            raise ValueError(f"grid {grid} is not a power-of-two multiple of {self.token_grid}")

        layers = [nn.Conv2d(in_ch, width, 1), nn.GroupNorm(32, width), nn.GELU()]
        channels = width
        for _ in range(ups):
            nxt = max(64, channels // 2)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, nxt, 3, padding=1), nn.GroupNorm(32, nxt), nn.GELU(),
            ]
            channels = nxt
        # One Sequential, stem included, to match the state dict older checkpoints hold.
        self.decoder = nn.Sequential(*layers)
        self.stem_len = 3
        if attention_layers:
            # Learned position embedding for the attention blocks.
            self.pos = nn.Parameter(torch.zeros(1, self.token_grid ** 2, width))
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.attention = nn.ModuleList(
            [AttentionBlock(width, heads) for _ in range(attention_layers)])
        if self.attention_skip:
            # So the decoder sees the local pre-attention features beside the attended ones.
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(width * 2, width, 1), nn.GroupNorm(32, width), nn.GELU())
        self.logit_head = nn.Conv2d(channels, 1, 1)
        self.offset_head = nn.Conv2d(channels, 2, 1)

    def features(self, pixel_values):
        return self.patch_token_map(pixel_values, self.token_grid, self.token_grid)[0]

    def forward(self, pixel_values):
        x = self.decoder[:self.stem_len](self.features(pixel_values))
        if len(self.attention):
            attended = attend(x, self.pos, self.attention)
            x = self.skip_fuse(torch.cat([x, attended], dim=1)) if self.attention_skip else attended
        x = self.decoder[self.stem_len:](x)
        return self.logit_head(x).squeeze(1), self.offset_head(x)


def target_map(cells, mask, grid, sigma):
    """Per-cell objectness target: the max of Gaussian bumps around each label."""
    axis = torch.arange(grid, device=cells.device, dtype=cells.dtype)
    # cell centres are at +0.5, and a label is a continuous cell coordinate
    dx = axis[None, None, :] - (cells[..., 0:1] - 0.5)
    dy = axis[None, None, :] - (cells[..., 1:2] - 0.5)
    gauss = torch.exp(-(dy[..., :, None] ** 2 + dx[..., None, :] ** 2) / (2 * sigma ** 2))
    return (gauss * mask[..., None, None]).amax(dim=1)


# Target below which a cell of an incomplete frame is unsupervised (a bit over 3 sigma).
BUMP_FLOOR = 0.01


def supervision_mask(target, complete, bump_floor=BUMP_FLOOR):
    """Cells this batch may train on: all of a complete frame, but only the label bumps of a
    teleop frame."""
    return torch.maximum(complete[:, None, None], (target >= bump_floor).to(target.dtype))


def objectness_loss(logits, offsets, points, mask, complete, image_size, grid,
                    offset_weight=1.0, cell_sigma=CELL_SIGMA, pos_weight=1.0):
    """Masked, pos_weight-balanced per-cell BCE plus L1 on each label's sub-cell offset."""
    scale = image_size / grid
    cells = points / scale
    # Only the target slots this batch uses.
    used = max(1, int(mask.sum(1).max().item()))
    cells, mask = cells[:, :used], mask[:, :used]

    target = target_map(cells, mask, grid, cell_sigma)
    supervised = supervision_mask(target, complete)
    weight = torch.where(target >= BUMP_FLOOR,
                         torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype),
                         torch.ones((), device=logits.device, dtype=logits.dtype))
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (per_cell * weight * supervised).sum() / (weight * supervised).sum().clamp_min(1.0)

    cx = cells[..., 0].floor().clamp(0, grid - 1).long()
    cy = cells[..., 1].floor().clamp(0, grid - 1).long()
    index = cy * grid + cx
    frac = cells - torch.stack([cx, cy], dim=-1).to(cells.dtype)
    # (B, 2, T) out of the gather, one offset pair per label, back to (B, T, 2).
    picked = offsets.flatten(2).gather(2, index[:, None, :].expand(-1, 2, -1)).permute(0, 2, 1)
    per_label = F.l1_loss(picked.sigmoid(), frac.clamp(0.0, 1.0), reduction="none").mean(-1)
    l1 = (per_label * mask).sum() / mask.sum().clamp_min(1.0)
    return bce + offset_weight * l1, bce.detach(), l1.detach()


# ==========================================
# SQUARE SYMMETRIES
# ==========================================
# The floor has no canonical orientation, so the 8 symmetries of the square are exact, free augmentation, used for TTA too.

def dihedral_image(img, t: int):
    """One of the 8 square symmetries applied to a CHW (or BCHW) tensor."""
    out = torch.rot90(img, t % 4, dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if t >= 4 else out


def dihedral_point(u, v, t: int, size: int):
    """The same symmetry applied to a point, in pixel-centre coordinates."""
    last = size - 1
    for _ in range(t % 4):
        u, v = v, last - u  # rot90 on dims (-2, -1) is counter-clockwise
    if t >= 4:
        u = last - u
    return u, v


def inverse_dihedral_map(m, t: int):
    if t >= 4:
        m = torch.flip(m, dims=(-1,))
    return torch.rot90(m, -(t % 4), dims=(-2, -1))


def decode(probs, offsets, image_size, grid, top_k=1, nms_radius=2):
    """Peak cells plus offsets, as (B, k, 2) pixel coordinates and (B, k) probabilities."""
    scale = image_size / grid
    prob = probs.view(-1, 1, grid, grid)

    if top_k > 1:
        prob = local_maxima(prob, nms_radius)

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    cx = (index % grid).float()
    cy = torch.div(index, grid, rounding_mode="floor").float()

    off = offsets.flatten(2).sigmoid()
    picked = torch.stack([
        off[:, 0].gather(1, index),
        off[:, 1].gather(1, index),
    ], dim=-1)
    uv = (torch.stack([cx, cy], dim=-1) + picked) * scale
    return uv, scores


def predict(model, images, tta=False, top_k=1):
    """Decoded predictions for a batch, optionally averaged over the 8 square symmetries."""
    logits, offsets = model(images)
    probs = logits.sigmoid()
    if tta:
        acc = probs
        for t in range(1, 8):
            transformed, _ = model(dihedral_image(images, t))
            acc = acc + inverse_dihedral_map(transformed.sigmoid(), t)
        probs = acc / 8
    return decode(probs, offsets, model.image_size, model.grid, top_k=top_k)


# ==========================================
# LIVE INFERENCE
# ==========================================

TARGETING_MODEL_REPOID = "naavox/targeting"
TARGETING_MODEL_FILENAME = "ortho_target.pth"


def prepare_ortho_image(rgb, image_size, device):
    return input_batch(rgb, (image_size, image_size), device)


@torch.no_grad()
def predict_room_targets(model, rgb, device, top_k=1, tta=False,
                         min_probability=TARGET_THRESHOLD):
    """Every target in one ortho frame above min_probability, as [(x_m, y_m, probability)]
    in descending order."""
    batch = prepare_ortho_image(rgb, model.image_size, device)
    uv, scores = predict(model, batch, tta=tta, top_k=top_k)
    uv, scores = uv[0].cpu().numpy(), scores[0].cpu().numpy()

    out = []
    for (u, v), score in zip(uv, scores):
        if score < min_probability:
            continue
        x_m, y_m = ortho_px_to_room(float(u), float(v), model.image_size, model.image_size)
        out.append((x_m, y_m, float(score)))
    return out


def checkpoint_head(checkpoint):
    """What this checkpoint's logits mean, "objectness" or "softmax", inferred from its
    metrics for old files."""
    if "head" in checkpoint:
        return checkpoint["head"]
    return "objectness" if "bce" in (checkpoint.get("metrics") or {}) else "softmax"


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint_head(checkpoint) != "objectness":
        raise ValueError(
            f"{path} predates the objectness head: its logits are one softmax over cells, "
            f"which this reads as independent sigmoids and saturates. Retrain it.")
    freeze = checkpoint.get("freeze", True)
    model = OrthoTargetNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        grid=checkpoint["grid"], fuse_layers=checkpoint["fuse_layers"], freeze=freeze,
        # absent from every checkpoint trained before the attention blocks existed
        attention_layers=checkpoint.get("attention_layers", 0),
        attention_skip=checkpoint.get("attention_skip", False),
    ).to(device)
    load_head_state(model, checkpoint)
    # The operating threshold travels with the checkpoint, with a fallback for old files.
    model.threshold = float(checkpoint.get("threshold", TARGET_THRESHOLD))
    model.eval()
    return model, checkpoint
