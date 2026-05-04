#!/usr/bin/env python
"""Train sequence-only localization baselines on official DeepLoc folds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import config_to_dict, load_config, make_run_id, parse_args
from esm_probe.data.deeploc import load_fold_split, load_localization_table
from esm_probe.evaluation.localization_metrics import compute_localization_metrics, tune_multilabel_thresholds
from esm_probe.models.baselines import featurize_sequences, make_baseline_classifier
from esm_probe.training.seeds import set_seed
from esm_probe.utils.io import write_json, write_table
from esm_probe.utils.logging import setup_logging
from esm_probe.utils.paths import make_run_dir, save_run_metadata


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    set_seed(cfg.project.seed, cfg.project.deterministic)
    setup_logging(cfg.logging.level)
    table, label_cols = load_localization_table(cfg)
    table["id"] = table["id"].astype(str)
    by_id = table.set_index("id")

    for method in cfg.baseline.methods:
        for fold in cfg.data.folds:
            train_ids, val_ids = load_fold_split(cfg, int(fold))
            payload = {"type": "baseline", "method": method, "fold": fold, "config": config_to_dict(cfg)}
            run_id = make_run_id(f"baseline-{method}-fold{fold}", payload)
            run_dir = make_run_dir(cfg, run_id)
            setup_logging(cfg.logging.level, Path(cfg.project.output_dir) / "logs" / f"{run_id}.log")
            save_run_metadata(cfg, run_dir)

            train = by_id.loc[train_ids]
            val = by_id.loc[val_ids]
            x_train = featurize_sequences(train["sequence"].tolist(), method)
            x_val = featurize_sequences(val["sequence"].tolist(), method)
            y_train = train[label_cols].to_numpy(dtype=int)
            y_val = val[label_cols].to_numpy(dtype=int)

            clf = make_baseline_classifier(cfg.baseline.class_weight, cfg.baseline.max_iter)
            clf.fit(x_train, y_train)
            probs = clf.predict_proba(x_val)
            if isinstance(probs, list):
                probs = np.vstack([p[:, 1] for p in probs]).T
            thresholds = tune_multilabel_thresholds(probs, y_val)
            metrics = compute_localization_metrics(probs, y_val, thresholds, label_cols)
            metrics.update(
                {
                    "run_id": run_id,
                    "run_type": "baseline",
                    "dataset": cfg.data.dataset or "deeploc2",
                    "method": method,
                    "fold": int(fold),
                }
            )
            write_json(Path(cfg.project.output_dir) / "metrics" / f"{run_id}.json", metrics)
            write_json(run_dir / "metrics.json", metrics)

            pred = pd.DataFrame({"id": val_ids})
            for i, label in enumerate(label_cols):
                pred[f"prob_{label}"] = probs[:, i]
                pred[f"pred_{label}"] = (probs[:, i] >= thresholds[i]).astype(int)
                pred[f"true_{label}"] = y_val[:, i]
            write_table(Path(cfg.project.output_dir) / "predictions" / f"{run_id}.csv", pred)
            write_table(run_dir / "predictions.csv", pred)


if __name__ == "__main__":
    main()
