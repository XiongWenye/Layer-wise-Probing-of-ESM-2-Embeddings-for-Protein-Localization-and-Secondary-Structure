#!/usr/bin/env python
"""Train frozen-embedding localization probes."""

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
from esm_probe.data.datasets import LocalizationEmbeddingDataset, collate_token_embeddings
from esm_probe.data.deeploc import load_fold_split, load_localization_table
from esm_probe.evaluation.localization_metrics import compute_localization_metrics, sigmoid, tune_multilabel_thresholds
from esm_probe.models.esm_embedder import embedding_paths, resolve_device, validate_layers
from esm_probe.models.probes import LocalizationProbe
from esm_probe.training.loops import save_checkpoint, train_localization_probe_loop
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


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    device = resolve_device(cfg.project.device)
    dataset = cfg.data.dataset or "deeploc2"
    table, label_cols = load_localization_table(cfg, dataset)

    for esm_name in cfg.model.esm_names:
        layers = validate_layers(esm_name, cfg.model.layers, cfg.model.allow_invalid_layers)
        for layer, pooling, hidden_dim, dropout, lr, wd, seed, fold in itertools.product(
            layers,
            cfg.probe.pooling,
            cfg.probe.hidden_dim,
            cfg.probe.dropout,
            _as_list(cfg.training.learning_rate),
            _as_list(cfg.training.weight_decay),
            cfg.training.seeds,
            cfg.data.folds,
        ):
            set_seed(int(seed), cfg.project.deterministic)
            h5_path, manifest_path = embedding_paths(cfg.data.processed_dir, dataset, esm_name, int(layer))
            if not h5_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"Missing embeddings for {esm_name} layer {layer}: {h5_path}")
            train_ids, val_ids = load_fold_split(cfg, int(fold))
            train_ds = LocalizationEmbeddingDataset(table, label_cols, h5_path, manifest_path, train_ids)
            val_ds = LocalizationEmbeddingDataset(table, label_cols, h5_path, manifest_path, val_ids)
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.project.num_workers,
                collate_fn=collate_token_embeddings,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.project.num_workers,
                collate_fn=collate_token_embeddings,
            )
            input_dim = _embedding_dim(h5_path)
            model = LocalizationProbe(
                input_dim=input_dim,
                num_labels=len(label_cols),
                pooling=str(pooling),
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                activation=cfg.probe.activation,
            )
            payload = {
                "type": "probe",
                "esm_name": esm_name,
                "layer": int(layer),
                "pooling": pooling,
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(wd),
                "seed": int(seed),
                "fold": int(fold),
                "config": config_to_dict(cfg),
            }
            run_id = make_run_id(f"probe-l{layer}-{pooling}-fold{fold}-seed{seed}", payload)
            run_dir = make_run_dir(cfg, run_id)
            setup_logging(cfg.logging.level, Path(cfg.project.output_dir) / "logs" / f"{run_id}.log")
            save_run_metadata(cfg, run_dir)
            result = train_localization_probe_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg.training.epochs,
                patience=cfg.training.patience,
                learning_rate=float(lr),
                weight_decay=float(wd),
                gradient_clip_norm=cfg.training.gradient_clip_norm,
                device=device,
                amp=cfg.training.amp,
            )
            model.load_state_dict(result.best_state)
            logits = result.val_logits.numpy()
            probs = sigmoid(logits)
            targets = result.val_targets.numpy().astype(int)
            thresholds = tune_multilabel_thresholds(probs, targets)
            metrics = compute_localization_metrics(probs, targets, thresholds, label_cols)
            metrics.update(
                {
                    "run_id": run_id,
                    "run_type": "probe",
                    "dataset": dataset,
                    "esm_name": esm_name,
                    "layer": int(layer),
                    "pooling": str(pooling),
                    "hidden_dim": int(hidden_dim),
                    "dropout": float(dropout),
                    "learning_rate": float(lr),
                    "weight_decay": float(wd),
                    "seed": int(seed),
                    "fold": int(fold),
                    "best_epoch": int(result.best_epoch),
                }
            )
            write_json(Path(cfg.project.output_dir) / "metrics" / f"{run_id}.json", metrics)
            write_json(run_dir / "metrics.json", metrics)
            result.history.to_csv(run_dir / "training_curve.csv", index=False)
            save_checkpoint(
                run_dir / "checkpoint.pt",
                model,
                {
                    "run_id": run_id,
                    "label_cols": label_cols,
                    "thresholds": thresholds.tolist(),
                    "payload": payload,
                },
            )
            pred = pd.DataFrame({"id": result.val_ids})
            for i, label in enumerate(label_cols):
                pred[f"prob_{label}"] = probs[:, i]
                pred[f"pred_{label}"] = (probs[:, i] >= thresholds[i]).astype(int)
                pred[f"true_{label}"] = targets[:, i]
            write_table(Path(cfg.project.output_dir) / "predictions" / f"{run_id}.csv", pred)
            write_table(run_dir / "predictions.csv", pred)


if __name__ == "__main__":
    main()
