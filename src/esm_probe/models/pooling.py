"""Masked sequence pooling layers."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """A single-head learned attention pooling layer."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class MaskedPooling(nn.Module):
    """Mean, max, or attention pooling over residue embeddings."""

    def __init__(self, mode: str, input_dim: int) -> None:
        super().__init__()
        self.mode = mode
        self.attention = AttentionPooling(input_dim) if mode == "attention" else None
        if mode not in {"mean", "max", "attention"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
            return (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
        if self.mode == "max":
            masked = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
            return masked.max(dim=1).values
        assert self.attention is not None
        return self.attention(x, mask)
