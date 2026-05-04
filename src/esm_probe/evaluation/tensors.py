"""Tensor helpers for evaluation loops."""

from __future__ import annotations

import torch


def cat_padded_batch_tensors(tensors: list[torch.Tensor], pad_value: int | float | bool = 0) -> torch.Tensor:
    """Concatenate batch-first sequence tensors with different padded lengths."""

    if not tensors:
        raise ValueError("Cannot concatenate an empty tensor list")
    max_len = max(tensor.shape[1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[1]
        if pad_len > 0:
            pad_shape = (tensor.shape[0], pad_len, *tensor.shape[2:])
            pad = tensor.new_full(pad_shape, pad_value)
            tensor = torch.cat((tensor, pad), dim=1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)
