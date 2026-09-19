"""Layers shared by the models that read DINO patch tokens as a grid of cells."""

import torch
import torch.nn.functional as F
from torch import nn


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
