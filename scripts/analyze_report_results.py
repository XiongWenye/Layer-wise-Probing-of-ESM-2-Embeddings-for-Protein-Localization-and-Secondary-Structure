#!/usr/bin/env python
"""Create report-facing diagnostic tables from saved metrics and predictions.

This script is intentionally light-weight: it reads existing JSON/CSV artifacts
and does not run ESM extraction, training, or evaluation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
TABLES = RESULTS / "tables"

LOCALIZATION_LABELS = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def _metric_files() -> list[tuple[Path, dict[str, object]]]:
    payloads = []
    for path in sorted(METRICS.glob("*.json")):
        payloads.append((path, _load_json(path)))
    return payloads


def _format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def _model_label(metric: dict[str, object]) -> str:
    esm_name = str(metric.get("esm_name", "ESM-2"))
    scale = "150M" if "150M" in esm_name else "35M"
    layer = metric.get("layer")
    pooling = metric.get("pooling")
    if layer is not None and pooling is not None:
        return f"{scale} L{int(layer)} {pooling}"
    if layer is not None:
        return f"{scale} L{int(layer)}"
    return str(metric.get("run_id", "run"))


def _prediction_path(run_id: str) -> Path:
    return PREDICTIONS / f"{run_id}.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_table(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _localization_transfer(metrics: list[tuple[Path, dict[str, object]]]) -> list[dict[str, object]]:
    by_run_id = {str(payload.get("run_id")): payload for _, payload in metrics}
    rows = []
    for _, payload in metrics:
        if payload.get("run_type") != "external_eval" or str(payload.get("dataset", "")).lower() != "hpa":
            continue
        selected_run_id = str(payload.get("selected_run_id", ""))
        selected = by_run_id.get(selected_run_id, {})
        rows.append(
            {
                "model": _model_label(payload),
                "selected_run_id": selected_run_id,
                "deeploc_cv_macro_f1": _format_float(selected.get("macro_f1")),
                "hpa_macro_f1": _format_float(payload.get("macro_f1")),
                "hpa_micro_f1": _format_float(payload.get("micro_f1")),
                "hpa_jaccard": _format_float(payload.get("jaccard")),
            }
        )
    return sorted(rows, key=lambda row: row["model"])


def _best_hpa_metric(metrics: list[tuple[Path, dict[str, object]]]) -> dict[str, object]:
    external = [
        payload
        for _, payload in metrics
        if payload.get("run_type") == "external_eval" and str(payload.get("dataset", "")).lower() == "hpa"
    ]
    if not external:
        raise RuntimeError("No HPA external-eval metrics found.")
    return max(external, key=lambda item: float(item.get("macro_f1", float("-inf"))))


def _hpa_label_diagnostics(metric: dict[str, object]) -> list[dict[str, object]]:
    rows = _read_rows(_prediction_path(str(metric["run_id"])))
    mcc_by_label = metric.get("per_label_mcc") or {}
    diagnostics = []
    for label in LOCALIZATION_LABELS:
        true_count = sum(int(row[f"true_{label}"]) for row in rows)
        pred_count = sum(int(row[f"pred_{label}"]) for row in rows)
        tp = sum(int(row[f"true_{label}"]) and int(row[f"pred_{label}"]) for row in rows)
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / true_count if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        diagnostics.append(
            {
                "label": label,
                "true_count": true_count,
                "pred_count": pred_count,
                "true_prevalence": _format_float(true_count / len(rows)),
                "tp": tp,
                "f1": _format_float(f1),
                "mcc": _format_float(mcc_by_label.get(label)),
            }
        )
    return diagnostics


def _cb513_sequence_bins() -> list[dict[str, object]]:
    eval_metrics = [
        payload
        for _, payload in _metric_files()
        if payload.get("run_type") == "secondary_external_eval"
        and str(payload.get("dataset", "")).lower() == "cb513"
    ]
    if not eval_metrics:
        raise RuntimeError("No CB513 secondary external-eval metric found.")
    selected = eval_metrics[0]
    rows = _read_rows(_prediction_path(str(selected["run_id"])))
    bins = [
        ("short (<=100 aa)", 0, 100),
        ("medium (101-200 aa)", 101, 200),
        ("long (>200 aa)", 201, float("inf")),
    ]
    summaries = []
    for name, lower, upper in bins:
        group = [
            row
            for row in rows
            if int(row["length"]) >= lower and (upper == float("inf") or int(row["length"]) <= upper)
        ]
        if not group:
            continue
        q3_values = [float(row["q3_accuracy"]) for row in group]
        summaries.append(
            {
                "length_bin": name,
                "n_sequences": len(group),
                "mean_q3": _format_float(sum(q3_values) / len(q3_values)),
                "min_q3": _format_float(min(q3_values)),
                "max_q3": _format_float(max(q3_values)),
            }
        )
    q3_values = [float(row["q3_accuracy"]) for row in rows]
    summaries.append(
        {
            "length_bin": "all",
            "n_sequences": len(rows),
            "mean_q3": _format_float(sum(q3_values) / len(q3_values)),
            "min_q3": _format_float(min(q3_values)),
            "max_q3": _format_float(max(q3_values)),
        }
    )
    return summaries


def main() -> None:
    metrics = _metric_files()
    transfer = _localization_transfer(metrics)
    _write_table(
        TABLES / "report_localization_transfer.csv",
        [
            "model",
            "selected_run_id",
            "deeploc_cv_macro_f1",
            "hpa_macro_f1",
            "hpa_micro_f1",
            "hpa_jaccard",
        ],
        transfer,
    )

    hpa_metric = _best_hpa_metric(metrics)
    hpa_diagnostics = _hpa_label_diagnostics(hpa_metric)
    _write_table(
        TABLES / "report_hpa_label_diagnostics.csv",
        ["label", "true_count", "pred_count", "true_prevalence", "tp", "f1", "mcc"],
        hpa_diagnostics,
    )

    secondary_bins = _cb513_sequence_bins()
    _write_table(
        TABLES / "report_cb513_length_bins.csv",
        ["length_bin", "n_sequences", "mean_q3", "min_q3", "max_q3"],
        secondary_bins,
    )

    manifest = {
        "localization_transfer_rows": len(transfer),
        "hpa_label_diagnostic_model": _model_label(hpa_metric),
        "hpa_label_diagnostic_run_id": hpa_metric.get("run_id"),
        "cb513_length_bins": len(secondary_bins),
    }
    with (TABLES / "report_analysis_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
