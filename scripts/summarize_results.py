#!/usr/bin/env python
"""Summarize paper-facing metrics and select the final external-eval checkpoint."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import load_config, parse_args, stable_hash
from esm_probe.evaluation.reports import load_metric_files
from esm_probe.utils.io import ensure_dir, write_json, write_table
from esm_probe.utils.logging import setup_logging


TARGET_ESM = "esm2_t12_35M_UR50D"
TARGET_L30_ESM = "esm2_t30_150M_UR50D"
TARGET_LAYERS = [6, 12]
TARGET_POOLING = ["mean", "max", "attention"]
TARGET_FOLDS = [0, 1, 2, 3, 4]
TARGET_SEEDS = [1, 2, 3]
TARGET_BASELINES = ["aac", "dipeptide", "length_aac"]
TARGET_PROBE_FAMILIES = [
    (TARGET_ESM, layer, pooling)
    for layer in TARGET_LAYERS
    for pooling in TARGET_POOLING
] + [
    (TARGET_L30_ESM, 30, "attention"),
]


def _logical_metric_key(row: pd.Series) -> tuple[object, ...]:
    run_type = row.get("run_type")
    if run_type == "baseline":
        return (run_type, row.get("dataset"), row.get("method"), row.get("fold"))
    if run_type == "probe":
        return (
            run_type,
            row.get("dataset"),
            row.get("esm_name"),
            row.get("layer"),
            row.get("pooling"),
            row.get("fold"),
            row.get("seed"),
            row.get("hidden_dim"),
            row.get("dropout"),
            row.get("learning_rate"),
            row.get("weight_decay"),
        )
    if run_type == "secondary_probe":
        return (
            run_type,
            row.get("esm_name"),
            row.get("layer"),
            row.get("seed"),
            row.get("hidden_dim"),
            row.get("dropout"),
            row.get("lr"),
            row.get("weight_decay"),
        )
    if run_type == "secondary_baseline":
        return (
            run_type,
            row.get("dataset"),
            row.get("method"),
            row.get("seed"),
        )
    if run_type == "secondary_baseline_external_eval":
        return (
            run_type,
            row.get("dataset"),
            row.get("method"),
        )
    if run_type in {"external_eval", "secondary_external_eval"}:
        return (run_type, row.get("dataset"), row.get("selected_run_id"))
    return (run_type, row.get("run_id"), row.get("metric_file"))


def _deduplicate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frame = metrics.copy()
    frame["_logical_key"] = frame.apply(_logical_metric_key, axis=1)
    for col in ("macro_f1", "micro_f1", "jaccard", "q3_accuracy"):
        if col not in frame.columns:
            frame[col] = float("-inf")
    frame = frame.sort_values(
        ["_logical_key", "macro_f1", "micro_f1", "jaccard", "q3_accuracy", "metric_file"],
        na_position="first",
    )
    return frame.drop_duplicates("_logical_key", keep="last").drop(columns=["_logical_key"])


def _target_probe_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "run_type" not in metrics:
        return metrics.iloc[0:0].copy()
    rows = metrics[metrics["run_type"].eq("probe")].copy()
    rows = rows.assign(layer=lambda frame: frame["layer"].astype(int))
    family_index = pd.MultiIndex.from_tuples(TARGET_PROBE_FAMILIES, names=["esm_name", "layer", "pooling"])
    row_index = pd.MultiIndex.from_frame(rows[["esm_name", "layer", "pooling"]])
    return rows[
        row_index.isin(family_index)
        & rows["fold"].isin(TARGET_FOLDS)
        & rows["seed"].isin(TARGET_SEEDS)
        & rows["hidden_dim"].eq(128)
        & rows["dropout"].eq(0.1)
        & rows["learning_rate"].eq(0.001)
        & rows["weight_decay"].eq(0.0)
    ].copy()


def _summarize(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["n"])
    pieces = []
    grouped = frame.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(group))
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        pieces.append(row)
    return pd.DataFrame(pieces).sort_values(group_cols).reset_index(drop=True)


def _expected_probe_keys() -> set[tuple[object, ...]]:
    return {
        (esm_name, layer, pooling, fold, seed)
        for esm_name, layer, pooling in TARGET_PROBE_FAMILIES
        for fold in TARGET_FOLDS
        for seed in TARGET_SEEDS
    }


def _completed_probe_keys(probes: pd.DataFrame) -> set[tuple[object, ...]]:
    if probes.empty:
        return set()
    return set(probes[["esm_name", "layer", "pooling", "fold", "seed"]].itertuples(index=False, name=None))


def _select_family_and_run(probes: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object] | None:
    if probes.empty or summary.empty:
        return None
    complete = summary[summary["n"].eq(len(TARGET_FOLDS) * len(TARGET_SEEDS))]
    candidates = complete if not complete.empty else summary
    family = candidates.sort_values(["macro_f1_mean", "micro_f1_mean", "jaccard_mean"], ascending=False).iloc[0]
    family_rows = probes[
        probes["esm_name"].eq(family["esm_name"])
        & probes["layer"].eq(family["layer"])
        & probes["pooling"].eq(family["pooling"])
    ]
    selected = family_rows.sort_values(["macro_f1", "micro_f1", "jaccard"], ascending=False).iloc[0]
    return {
        "selected_run_id": selected["run_id"],
        "selection_rule": "highest validation Macro-F1 within best complete layer/pooling family",
        "family": {
            "esm_name": str(family["esm_name"]),
            "layer": int(family["layer"]),
            "pooling": str(family["pooling"]),
            "n": int(family["n"]),
            "macro_f1_mean": float(family["macro_f1_mean"]),
            "macro_f1_sd": float(family["macro_f1_sd"]),
            "micro_f1_mean": float(family["micro_f1_mean"]),
            "jaccard_mean": float(family["jaccard_mean"]),
        },
        "checkpoint_metric": {
            "macro_f1": float(selected["macro_f1"]),
            "micro_f1": float(selected["micro_f1"]),
            "jaccard": float(selected["jaccard"]),
            "fold": int(selected["fold"]),
            "seed": int(selected["seed"]),
        },
    }


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    results = Path(cfg.project.output_dir)
    tables_dir = ensure_dir(results / "tables")

    raw = load_metric_files(results / "metrics")
    metrics = _deduplicate_metrics(raw)
    probes = _target_probe_rows(metrics)
    probe_summary = _summarize(probes, ["esm_name", "layer", "pooling"], ["macro_f1", "micro_f1", "jaccard"])
    write_table(tables_dir / "localization_probe_summary.csv", probe_summary)

    baselines = metrics[metrics.get("run_type", pd.Series(dtype=object)).eq("baseline")].copy()
    baselines = baselines[baselines["method"].isin(TARGET_BASELINES) & baselines["fold"].isin(TARGET_FOLDS)]
    baseline_summary = _summarize(baselines, ["method"], ["macro_f1", "micro_f1", "jaccard"])
    write_table(tables_dir / "localization_baseline_summary.csv", baseline_summary)

    secondary = metrics[metrics.get("run_type", pd.Series(dtype=object)).eq("secondary_probe")].copy()
    if not secondary.empty:
        secondary["layer"] = secondary["layer"].astype(int)
    secondary_summary = _summarize(secondary, ["esm_name", "layer"], ["q3_accuracy"])
    write_table(tables_dir / "secondary_probe_summary.csv", secondary_summary)

    secondary_baselines = metrics[
        metrics.get("run_type", pd.Series(dtype=object)).eq("secondary_baseline")
    ].copy()
    secondary_baseline_summary = _summarize(
        secondary_baselines,
        ["method"],
        ["q3_accuracy"],
    )
    write_table(tables_dir / "secondary_baseline_summary.csv", secondary_baseline_summary)

    secondary_baseline_external = metrics[
        metrics.get("run_type", pd.Series(dtype=object)).eq("secondary_baseline_external_eval")
    ].copy()
    secondary_baseline_external_summary = _summarize(
        secondary_baseline_external,
        ["method"],
        ["q3_accuracy"],
    )
    write_table(
        tables_dir / "secondary_baseline_external_summary.csv",
        secondary_baseline_external_summary,
    )

    completed = _completed_probe_keys(probes)
    expected = _expected_probe_keys()
    missing = sorted(expected - completed, key=lambda item: (item[0], item[1], item[2], item[3]))
    selection = _select_family_and_run(probes, probe_summary)
    manifest = {
        "config_hash": stable_hash(cfg.model_dump(mode="json")),
        "num_metric_rows_raw": int(len(raw)),
        "num_metric_rows_deduplicated": int(len(metrics)),
        "localization_probe_target_rows": len(expected),
        "localization_probe_completed_rows": len(completed),
        "localization_probe_missing_rows": [
            {"esm_name": esm_name, "layer": layer, "pooling": pooling, "fold": fold, "seed": seed}
            for esm_name, layer, pooling, fold, seed in missing
        ],
        "baseline_target_rows": len(TARGET_BASELINES) * len(TARGET_FOLDS),
        "baseline_completed_rows": int(len(baselines)),
        "secondary_probe_rows": int(len(secondary)),
        "secondary_baseline_rows": int(len(secondary_baselines)),
        "secondary_baseline_external_rows": int(len(secondary_baseline_external)),
        "hpa_selection": selection,
    }
    write_json(tables_dir / "summary_manifest.json", manifest)


if __name__ == "__main__":
    main()
