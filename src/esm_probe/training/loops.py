"""Training and prediction loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from esm_probe.utils.io import ensure_dir


@dataclass
class TrainResult:
    """Outputs from a probe training run."""

    best_state: dict[str, torch.Tensor]
    best_epoch: int
    history: pd.DataFrame
    val_logits: torch.Tensor
    val_targets: torch.Tensor
    val_ids: list[str]


def train_localization_probe_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: torch.device,
    amp: bool,
) -> TrainResult:
    """Train a sequence-level multilabel probe with early stopping on validation loss."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    model.to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for x, mask, y, _ids in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x, mask)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * x.shape[0]
            seen += x.shape[0]

        val_loss, _, _, _ = predict_localization(model, val_loader, device, return_loss=True)
        train_loss = train_loss / max(seen, 1)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    _, logits, targets, ids = predict_localization(model, val_loader, device, return_loss=True)
    return TrainResult(best_state, best_epoch, pd.DataFrame(rows), logits, targets, ids)


def predict_localization(
    model: nn.Module,
    loader,
    device: torch.device,
    return_loss: bool = False,
) -> tuple[float, torch.Tensor, torch.Tensor, list[str]]:
    """Predict sequence-level multilabel logits."""

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    ids_all: list[str] = []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, mask, y, ids in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            if return_loss:
                loss_sum += float(criterion(logits, y).detach().cpu())
            n += x.shape[0]
            logits_all.append(logits.detach().cpu())
            targets_all.append(y.detach().cpu())
            ids_all.extend(ids)
    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    return loss_sum / max(n, 1), logits_cat, targets_cat, ids_all


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict) -> None:
    """Save a model checkpoint."""

    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
