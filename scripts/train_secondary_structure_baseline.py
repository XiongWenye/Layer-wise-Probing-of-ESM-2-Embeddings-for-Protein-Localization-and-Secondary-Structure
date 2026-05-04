#!/usr/bin/env python
"""Train deterministic secondary-structure baselines and evaluate on CB513."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.baselines.secondary_structure import (
    evaluate_secondary_baseline,
    fit_secondary_baseline,
)
from esm_probe.config import config_to_dict, load_config, make_run_id, parse_args
from esm_probe.constants import Q3_LABELS
from esm_probe.data.secondary_structure import load_secondary_table
from esm_probe.training.seeds import set_seed
from esm_probe.utils.io import write_json, write_table
from esm_probe.utils.logging import setup_logging
from esm_probe.utils.paths import make_run_dir, save_run_metadata


def _split_train_val(ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids_array = np.array(ids)
    order = rng.permutation(len(ids_array))
    cutoff = max(1, int(0.9 * len(ids_array)))
    train_ids = ids_array[order[:cutoff]].tolist()
    val_ids = ids_array[order[cutoff:]].tolist()
    if not val_ids:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]
    return train_ids, val_ids


def _write_result(cfg, run_id: str, metrics: dict[str, object], predictions, payload: dict[str, object]) -> None:
    run_dir = make_run_dir(cfg, run_id)
    setup_logging(cfg.logging.level, Path(cfg.project.output_dir) / "logs" / f"{run_id}.log")
    save_run_metadata(cfg, run_dir)
    write_json(Path(cfg.project.output_dir) / "metrics" / f"{run_id}.json", metrics)
    write_json(run_dir / "metrics.json", metrics)
    write_table(Path(cfg.project.output_dir) / "predictions" / f"{run_id}.csv", predictions)
    write_table(run_dir / "predictions.csv", predictions)
    write_json(run_dir / "model.json", payload)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    train_dataset = cfg.data.train_dataset or "cullpdb5926_filtered"
    final_dataset = cfg.data.final_test_dataset or "cb513"
    train_table = load_secondary_table(cfg, train_dataset)
    final_table = load_secondary_table(cfg, final_dataset)
    all_train_ids = train_table["id"].astype(str).tolist()
    methods = cfg.baseline.methods

    for method in methods:
        for seed in cfg.training.seeds:
            set_seed(int(seed), cfg.project.deterministic)
            train_ids, val_ids = _split_train_val(all_train_ids, int(seed))
            model = fit_secondary_baseline(train_table, train_ids, str(method), Q3_LABELS)
            metrics, predictions = evaluate_secondary_baseline(train_table, val_ids, model, Q3_LABELS)
            payload = {
                "type": "secondary_baseline",
                "method": str(method),
                "seed": int(seed),
                "train_dataset": train_dataset,
                "validation_dataset": train_dataset,
                "num_train_proteins": len(train_ids),
                "num_validation_proteins": len(val_ids),
                "global_label": model.global_label,
                "aa_to_label": model.aa_to_label,
                "config": config_to_dict(cfg),
            }
            run_id = make_run_id(f"secondary-baseline-{method}-seed{seed}", payload)
            metrics.update(
                {
                    "run_id": run_id,
                    "run_type": "secondary_baseline",
                    "dataset": train_dataset,
                    "method": str(method),
                    "seed": int(seed),
                    "global_label": model.global_label,
                }
            )
            _write_result(cfg, run_id, metrics, predictions, {k: v for k, v in payload.items() if k != "config"})

        model = fit_secondary_baseline(train_table, all_train_ids, str(method), Q3_LABELS)
        final_ids = final_table["id"].astype(str).tolist()
        metrics, predictions = evaluate_secondary_baseline(final_table, final_ids, model, Q3_LABELS)
        payload = {
            "type": "secondary_baseline_external_eval",
            "method": str(method),
            "train_dataset": train_dataset,
            "dataset": final_dataset,
            "num_train_proteins": len(all_train_ids),
            "num_eval_proteins": len(final_ids),
            "global_label": model.global_label,
            "aa_to_label": model.aa_to_label,
            "config": config_to_dict(cfg),
        }
        run_id = make_run_id(f"cb513-baseline-{method}", payload)
        metrics.update(
            {
                "run_id": run_id,
                "run_type": "secondary_baseline_external_eval",
                "dataset": final_dataset,
                "method": str(method),
                "global_label": model.global_label,
            }
        )
        _write_result(cfg, run_id, metrics, predictions, {k: v for k, v in payload.items() if k != "config"})


if __name__ == "__main__":
    main()
