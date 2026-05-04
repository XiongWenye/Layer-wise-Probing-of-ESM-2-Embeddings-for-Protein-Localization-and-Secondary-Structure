#!/usr/bin/env python
"""Evaluate selected localization probes on the untouched external HPA set."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import load_config, make_run_id, parse_args
from esm_probe.data.datasets import LocalizationEmbeddingDataset, collate_token_embeddings
from esm_probe.data.deeploc import load_localization_table
from esm_probe.evaluation.localization_metrics import compute_localization_metrics, sigmoid
from esm_probe.models.esm_embedder import embedding_paths, resolve_device
from esm_probe.models.probes import LocalizationProbe
from esm_probe.training.loops import predict_localization
from esm_probe.utils.io import read_json, write_json, write_table
from esm_probe.utils.logging import setup_logging
from esm_probe.utils.paths import make_run_dir, save_run_metadata


def _embedding_dim(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as handle:
        group = handle["embeddings"]
        first_key = next(iter(group.keys()))
        return int(group[first_key].shape[-1])


def _load_selected_metric(metrics_dir: Path, selected_run_id: str) -> dict:
    for path in metrics_dir.glob("*.json"):
        payload = read_json(path)
        if payload.get("run_id") == selected_run_id:
            payload["_path"] = str(path)
            if payload.get("run_type") != "probe":
                raise ValueError(f"Selected run is not a localization probe: {selected_run_id}")
            return payload
    raise FileNotFoundError(f"Selected run metric not found in {metrics_dir}: {selected_run_id}")


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    device = resolve_device(cfg.project.device)
    if args.selected_run_id is None:
        raise ValueError("Final localization evaluation requires --selected-run-id.")
    selected = _load_selected_metric(Path(cfg.project.output_dir) / "metrics", args.selected_run_id)
    original_run_id = selected["run_id"]
    run_dir = Path(cfg.project.output_dir) / cfg.project.run_group / original_run_id
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metadata = ckpt["metadata"]
    label_cols = metadata["label_cols"]
    thresholds = torch.tensor(metadata["thresholds"]).numpy()

    external = cfg.data.external_test or "hpa"
    table, _ = load_localization_table(cfg, external)
    ids = table["id"].astype(str).tolist()
    esm_name = selected["esm_name"]
    layer = int(selected["layer"])
    h5_path, manifest_path = embedding_paths(cfg.data.processed_dir, external, esm_name, layer)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Missing HPA embeddings: {h5_path}. Run extract_esm_embeddings.py for dataset {external}."
        )
    dataset = LocalizationEmbeddingDataset(table, label_cols, h5_path, manifest_path, ids)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.project.num_workers,
        collate_fn=collate_token_embeddings,
    )
    model = LocalizationProbe(
        input_dim=_embedding_dim(h5_path),
        num_labels=len(label_cols),
        pooling=str(selected["pooling"]),
        hidden_dim=int(selected["hidden_dim"]),
        dropout=float(selected["dropout"]),
        activation=cfg.probe.activation,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    _, logits, targets, pred_ids = predict_localization(model, loader, device, return_loss=True)
    probs = sigmoid(logits.numpy())
    y = targets.numpy().astype(int)
    metrics = compute_localization_metrics(probs, y, thresholds, label_cols)
    run_id = make_run_id(f"hpa-eval-{original_run_id}", {"selected": selected, "external": external})
    out_dir = make_run_dir(cfg, run_id)
    setup_logging(cfg.logging.level, Path(cfg.project.output_dir) / "logs" / f"{run_id}.log")
    save_run_metadata(cfg, out_dir)
    metrics.update(
        {
            "run_id": run_id,
            "run_type": "external_eval",
            "dataset": external,
            "selected_run_id": original_run_id,
            "esm_name": esm_name,
            "layer": layer,
            "pooling": selected["pooling"],
        }
    )
    write_json(Path(cfg.project.output_dir) / "metrics" / f"{run_id}.json", metrics)
    write_json(out_dir / "metrics.json", metrics)
    pred = pd.DataFrame({"id": pred_ids})
    for i, label in enumerate(label_cols):
        pred[f"prob_{label}"] = probs[:, i]
        pred[f"pred_{label}"] = (probs[:, i] >= thresholds[i]).astype(int)
        pred[f"true_{label}"] = y[:, i]
    write_table(Path(cfg.project.output_dir) / "predictions" / f"{run_id}.csv", pred)
    write_table(out_dir / "predictions.csv", pred)


if __name__ == "__main__":
    main()
