"""Layers, cell operations and losses shared by the models that read DINO patch tokens as a
grid of cells."""

import torch
import torch.nn.functional as F
from torch import nn

# Cells either side of the winner whose softmax centre of mass gives the sub-cell position.
CENTROID_RADIUS = 2


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


def attend(x, pos, blocks):
    """Run attention blocks over a (B, C, H, W) map as tokens, with learned position
    embedding `pos`."""
    batch, channels, rows, cols = x.shape
    seq = x.flatten(2).transpose(1, 2) + pos
    for block in blocks:
        seq = block(seq)
    return seq.transpose(1, 2).reshape(batch, channels, rows, cols)


def local_maxima(prob, radius):
    pooled = F.max_pool2d(prob, radius * 2 + 1, stride=1, padding=radius)
    return torch.where(prob >= pooled, prob, torch.zeros_like(prob))


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


# -- losses ------------------------------------------------------------------


def masked_mean(values, mask):
    """Mean over rows weighted by `mask`; zero when there are none."""
    total = mask.sum()
    return (values * mask).sum() / total.clamp(min=1.0), total


def soft_cell_target(cell, grid, sigma):
    """A normalized Gaussian over the cell grid centred on the true position."""
    rows, cols = grid
    xs = torch.arange(cols, device=cell.device, dtype=cell.dtype).view(1, 1, cols) + 0.5
    ys = torch.arange(rows, device=cell.device, dtype=cell.dtype).view(1, rows, 1) + 0.5
    squared = ((xs - cell[:, 0].view(-1, 1, 1)) ** 2
               + (ys - cell[:, 1].view(-1, 1, 1)) ** 2)
    target = torch.exp(-0.5 * squared / (sigma * sigma))
    return target.flatten(1) / target.flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12)


def cell_loss(logits, cell, grid, index, sigma):
    """KL divergence of the cell head against a hard or softened target."""
    if sigma <= 0:
        return F.cross_entropy(logits.flatten(1), index, reduction="none")
    target = soft_cell_target(cell, grid, sigma)
    log_probs = F.log_softmax(logits.flatten(1), dim=1)
    entropy = -(target * target.clamp(min=1e-12).log()).sum(dim=1)
    return -(target * log_probs).sum(dim=1) - entropy
