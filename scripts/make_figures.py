#!/usr/bin/env python
"""Generate figures from saved metrics and training curves."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import load_config, parse_args, stable_hash
from esm_probe.evaluation.reports import load_metric_files
from esm_probe.utils.io import read_json
from esm_probe.utils.io import write_json
from esm_probe.utils.logging import setup_logging
from esm_probe.visualization.curves import load_training_curves
from esm_probe.visualization.figures import (
    plot_hpa_external_comparison,
    plot_layer_pooling_heatmap,
    plot_macro_f1_by_layer,
    plot_per_label_mcc_matrix,
    plot_training_curves,
)


def _logical_metric_key(row: pd.Series) -> tuple[object, ...]:
    """Group repeated executions of the same logical experiment."""

    run_type = row.get("run_type")
    if run_type == "baseline":
        return (
            run_type,
            row.get("dataset"),
            row.get("method"),
            row.get("fold"),
        )
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
    if run_type == "external_eval":
        return (
            run_type,
            row.get("dataset"),
            row.get("selected_run_id"),
        )
    return (run_type, row.get("run_id"), row.get("metric_file"))


def _deduplicate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate metric files while preserving the best completed row."""

    if metrics.empty:
        return metrics
    frame = metrics.copy()
    frame["_logical_key"] = frame.apply(_logical_metric_key, axis=1)
    for col in ("macro_f1", "micro_f1", "jaccard"):
        if col not in frame.columns:
            frame[col] = float("-inf")
    frame = frame.sort_values(
        ["_logical_key", "macro_f1", "micro_f1", "jaccard", "metric_file"],
        na_position="first",
    )
    frame = frame.drop_duplicates("_logical_key", keep="last")
    return frame.drop(columns=["_logical_key"]).reset_index(drop=True)


def _metric_payloads(metrics_dir: Path, metric_files: set[str]) -> list[dict[str, object]]:
    payloads = []
    for path in sorted(metrics_dir.glob("*.json")):
        if str(path) not in metric_files:
            continue
        payload = read_json(path)
        payloads.append({"metric_file": str(path), **payload})
    return payloads


def _mcc_run_label(payload: dict[str, object]) -> str:
    dataset = str(payload.get("dataset", "dataset"))
    run_type = payload.get("run_type")
    if run_type == "baseline":
        method = str(payload.get("method", "baseline")).upper()
        return f"{dataset} {method}"
    if payload.get("layer") is not None:
        layer = int(payload["layer"])
        pooling = payload.get("pooling", "")
        return f"{dataset} layer {layer} {pooling}".strip()
    return str(payload.get("run_id", "run"))


def _collect_mcc_rows(payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for label, value in (payload.get("per_label_mcc") or {}).items():
            rows.append({"run_label": _mcc_run_label(payload), "label": label, "mcc": value})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    results = Path(cfg.project.output_dir)
    figures = results / "figures"
    raw_metrics = load_metric_files(results / "metrics")
    metrics = _deduplicate_metrics(raw_metrics)
    payloads = _metric_payloads(results / "metrics", set(metrics["metric_file"]) if not metrics.empty else set())
    curves = load_training_curves(results)
    plot_layer_pooling_heatmap(metrics, figures / "layer_pooling_heatmap.png")
    plot_macro_f1_by_layer(metrics, figures / "macro_f1_by_layer.png")
    plot_hpa_external_comparison(metrics, figures / "hpa_external_comparison.png")
    plot_per_label_mcc_matrix(_collect_mcc_rows(payloads), figures / "per_label_mcc_matrix.png")
    plot_training_curves(curves, figures / "training_curves.png")
    write_json(
        figures / "figure_manifest.json",
        {
            "config_hash": stable_hash(cfg.model_dump(mode="json")),
            "num_metric_rows_raw": int(len(raw_metrics)),
            "num_metric_rows": int(len(metrics)),
            "num_duplicate_metric_rows": int(len(raw_metrics) - len(metrics)),
            "num_curve_rows": int(len(curves)),
        },
    )


if __name__ == "__main__":
    main()
