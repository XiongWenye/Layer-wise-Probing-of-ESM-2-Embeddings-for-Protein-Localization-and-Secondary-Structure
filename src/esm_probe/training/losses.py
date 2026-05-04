"""Loss helpers."""

from __future__ import annotations

import torch
from torch import nn


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over valid residue positions only."""

    loss = nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    loss = loss * mask.to(loss.dtype)
    return loss.sum() / mask.sum().clamp_min(1)
