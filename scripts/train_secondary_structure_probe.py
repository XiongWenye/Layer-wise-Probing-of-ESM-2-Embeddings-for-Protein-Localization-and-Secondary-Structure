#!/usr/bin/env python
"""Train residue-level secondary-structure probes on CullPDB-style data."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import config_to_dict, load_config, make_run_id, parse_args
from esm_probe.constants import Q3_LABELS
from esm_probe.data.datasets import SecondaryEmbeddingDataset, collate_secondary_embeddings
from esm_probe.data.secondary_structure import load_secondary_table
from esm_probe.evaluation.secondary_metrics import compute_secondary_metrics
from esm_probe.evaluation.tensors import cat_padded_batch_tensors
from esm_probe.models.esm_embedder import embedding_paths, resolve_device, validate_layers
from esm_probe.models.probes import TokenProbe
from esm_probe.training.losses import masked_cross_entropy
from esm_probe.training.seeds import set_seed
from esm_probe.utils.io import write_json, write_table
from esm_probe.utils.logging import setup_logging
from esm_probe.utils.paths import make_run_dir, save_run_metadata


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _embedding_dim(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as handle:
        group = handle["embeddings"]
        first_key = next(iter(group.keys()))
        return int(group[first_key].shape[-1])


def _split_train_val(ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    order = rng.permutation(len(ids))
    cutoff = max(1, int(0.9 * len(ids)))
    train_ids = ids[order[:cutoff]].tolist()
    val_ids = ids[order[cutoff:]].tolist()
    if not val_ids:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]
    return train_ids, val_ids


def _predict(model, loader, device):
    model.eval()
    logits_all, targets_all, masks_all, ids_all = [], [], [], []
    with torch.no_grad():
        for x, mask, y, ids in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x, mask).cpu()
            logits_all.append(logits)
            targets_all.append(y)
            masks_all.append(mask.cpu())
            ids_all.extend(ids)
    return (
        cat_padded_batch_tensors(logits_all),
        cat_padded_batch_tensors(targets_all),
        cat_padded_batch_tensors(masks_all, pad_value=False),
        ids_all,
    )


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    device = resolve_device(cfg.project.device)
    dataset_name = cfg.data.train_dataset or "cullpdb5926_filtered"
    table = load_secondary_table(cfg, dataset_name)
    label_to_index = {label: i for i, label in enumerate(Q3_LABELS)}
    for esm_name in cfg.model.esm_names:
        layers = validate_layers(esm_name, cfg.model.layers, cfg.model.allow_invalid_layers)
        for layer, hidden_dim, dropout, lr, wd, seed in itertools.product(
            layers,
            cfg.probe.hidden_dim,
            cfg.probe.dropout,
            _as_list(cfg.training.learning_rate),
            _as_list(cfg.training.weight_decay),
            cfg.training.seeds,
        ):
            set_seed(int(seed), cfg.project.deterministic)
            h5_path, manifest_path = embedding_paths(cfg.data.processed_dir, dataset_name, esm_name, int(layer))
            train_ids, val_ids = _split_train_val(table["id"].astype(str).tolist(), int(seed))
            train_ds = SecondaryEmbeddingDataset(table, h5_path, manifest_path, train_ids, label_to_index)
            val_ds = SecondaryEmbeddingDataset(table, h5_path, manifest_path, val_ids, label_to_index)
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.project.num_workers,
                collate_fn=collate_secondary_embeddings,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.project.num_workers,
                collate_fn=collate_secondary_embeddings,
            )
            model = TokenProbe(_embedding_dim(h5_path), len(Q3_LABELS), int(hidden_dim), float(dropout))
            model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
            best_loss = float("inf")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history = []
            stale = 0
            for epoch in range(1, cfg.training.epochs + 1):
                model.train()
                loss_sum = 0.0
                seen = 0
                for x, mask, y, _ids in train_loader:
                    x, mask, y = x.to(device), mask.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(x, mask)
                    loss = masked_cross_entropy(logits, y, mask)
                    loss.backward()
                    if cfg.training.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                    opt.step()
                    loss_sum += float(loss.detach().cpu()) * x.shape[0]
                    seen += x.shape[0]
                logits, targets, masks, _ = _predict(model, val_loader, device)
                val_loss = float(masked_cross_entropy(logits, targets, masks).detach().cpu())
                train_loss = loss_sum / max(seen, 1)
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.training.patience:
                        break
            model.load_state_dict(best_state)
            logits, targets, masks, ids = _predict(model, val_loader, device)
            preds = logits.argmax(dim=-1).numpy()
            metrics = compute_secondary_metrics(preds, targets.numpy(), masks.numpy(), Q3_LABELS)
            payload = {
                "type": "secondary_probe",
                "esm_name": esm_name,
                "layer": int(layer),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(wd),
                "seed": int(seed),
                "config": config_to_dict(cfg),
            }
            run_id = make_run_id(f"secondary-l{layer}-seed{seed}", payload)
            run_dir = make_run_dir(cfg, run_id)
            setup_logging(cfg.logging.level, Path(cfg.project.output_dir) / "logs" / f"{run_id}.log")
            save_run_metadata(cfg, run_dir)
            metrics.update({"run_id": run_id, "run_type": "secondary_probe", **{k: v for k, v in payload.items() if k != "config"}})
            write_json(Path(cfg.project.output_dir) / "metrics" / f"{run_id}.json", metrics)
            write_json(run_dir / "metrics.json", metrics)
            pd.DataFrame(history).to_csv(run_dir / "training_curve.csv", index=False)
            torch.save({"state_dict": model.state_dict(), "metadata": {"run_id": run_id, "label_names": Q3_LABELS, "payload": payload}}, run_dir / "checkpoint.pt")
            write_table(run_dir / "predictions.csv", pd.DataFrame({"id": ids}))


if __name__ == "__main__":
    main()
