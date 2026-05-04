#!/usr/bin/env python
"""Evaluate the selected secondary-structure probe on CB513."""

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
from esm_probe.constants import Q3_LABELS
from esm_probe.data.datasets import SecondaryEmbeddingDataset, collate_secondary_embeddings
from esm_probe.data.secondary_structure import load_secondary_table
from esm_probe.evaluation.secondary_metrics import compute_secondary_metrics
from esm_probe.evaluation.tensors import cat_padded_batch_tensors
from esm_probe.models.esm_embedder import embedding_paths, resolve_device
from esm_probe.models.probes import TokenProbe
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
            if payload.get("run_type") != "secondary_probe":
                raise ValueError(f"Selected run is not a secondary-structure probe: {selected_run_id}")
            return payload
    raise FileNotFoundError(f"Selected run metric not found in {metrics_dir}: {selected_run_id}")


def _q3_string(indices, mask) -> str:
    return "".join(Q3_LABELS[int(value)] for value, keep in zip(indices, mask) if bool(keep))


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    device = resolve_device(cfg.project.device)
    if args.selected_run_id is None:
        raise ValueError("Final secondary-structure evaluation requires --selected-run-id.")
    selected = _load_selected_metric(Path(cfg.project.output_dir) / "metrics", args.selected_run_id)
    selected_run_id = selected["run_id"]
    selected_dir = Path(cfg.project.output_dir) / cfg.project.run_group / selected_run_id
    ckpt_path = selected_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    final_dataset = cfg.data.final_test_dataset or "cb513"
    table = load_secondary_table(cfg, final_dataset)
    h5_path, manifest_path = embedding_paths(
        cfg.data.processed_dir,
        final_dataset,
        selected["esm_name"],
        int(selected["layer"]),
    )
    ids = table["id"].astype(str).tolist()
    label_to_index = {label: i for i, label in enumerate(Q3_LABELS)}
    dataset = SecondaryEmbeddingDataset(table, h5_path, manifest_path, ids, label_to_index)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.project.num_workers,
        collate_fn=collate_secondary_embeddings,
    )
    model = TokenProbe(
        _embedding_dim(h5_path),
        len(Q3_LABELS),
        int(selected["hidden_dim"]),
        float(selected["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    logits_all, targets_all, masks_all, ids_all = [], [], [], []
    with torch.no_grad():
        for x, mask, y, batch_ids in loader:
            logits_all.append(model(x.to(device), mask.to(device)).cpu())
            targets_all.append(y)
            masks_all.append(mask)
            ids_all.extend(batch_ids)
    logits = cat_padded_batch_tensors(logits_all)
    targets = cat_padded_batch_tensors(targets_all)
    masks = cat_padded_batch_tensors(masks_all, pad_value=False)
    preds = logits.argmax(dim=-1).numpy()
    metrics = compute_secondary_metrics(preds, targets.numpy(), masks.numpy(), Q3_LABELS)
    run_id = make_run_id(f"cb513-eval-{selected_run_id}", {"selected": selected, "dataset": final_dataset})
    out_dir = make_run_dir(cfg, run_id)
    setup_logging(cfg.logging.level, Path(cfg.project.output_dir) / "logs" / f"{run_id}.log")
    save_run_metadata(cfg, out_dir)
    metrics.update(
        {
            "run_id": run_id,
            "run_type": "secondary_external_eval",
            "dataset": final_dataset,
            "selected_run_id": selected_run_id,
            "esm_name": selected["esm_name"],
            "layer": int(selected["layer"]),
        }
    )
    write_json(Path(cfg.project.output_dir) / "metrics" / f"{run_id}.json", metrics)
    write_json(out_dir / "metrics.json", metrics)
    target_array = targets.numpy()
    mask_array = masks.numpy()
    pred_rows = []
    for i, protein_id in enumerate(ids_all):
        true_q3 = _q3_string(target_array[i], mask_array[i])
        pred_q3 = _q3_string(preds[i], mask_array[i])
        correct = sum(a == b for a, b in zip(true_q3, pred_q3))
        length = len(true_q3)
        pred_rows.append(
            {
                "id": protein_id,
                "length": length,
                "q3_accuracy": correct / max(length, 1),
                "true_q3": true_q3,
                "pred_q3": pred_q3,
            }
        )
    pred = pd.DataFrame(pred_rows)
    write_table(Path(cfg.project.output_dir) / "predictions" / f"{run_id}.csv", pred)
    write_table(out_dir / "predictions.csv", pred)


if __name__ == "__main__":
    main()
