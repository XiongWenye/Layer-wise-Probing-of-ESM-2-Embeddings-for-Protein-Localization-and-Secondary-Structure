"""Neural probes for frozen ESM embeddings."""

from __future__ import annotations

import torch
from torch import nn

from esm_probe.models.pooling import MaskedPooling


def activation_layer(name: str) -> nn.Module:
    """Return an activation module by name."""

    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class LocalizationProbe(nn.Module):
    """Sequence-level multilabel probe over token embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        pooling: str,
        hidden_dim: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.pooling = MaskedPooling(pooling, input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooling(x, mask)
        return self.head(pooled)


class TokenProbe(nn.Module):
    """Residue-level token classifier for secondary structure."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return self.head(x)
