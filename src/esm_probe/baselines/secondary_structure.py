"""Deterministic baselines for Q3 secondary-structure prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from esm_probe.evaluation.secondary_metrics import compute_secondary_metrics


@dataclass(frozen=True)
class SecondaryBaseline:
    """A fitted residue-level secondary-structure baseline."""

    method: str
    global_label: str
    aa_to_label: dict[str, str]


def _rows_by_id(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    table = frame.set_index("id")
    return table.loc[[str(seq_id) for seq_id in ids]]


def _most_common_label(counts: Counter[str], label_names: list[str]) -> str:
    """Return the most common label, breaking ties by label_names order."""

    return max(label_names, key=lambda label: (counts.get(label, 0), -label_names.index(label)))


def fit_secondary_baseline(
    frame: pd.DataFrame,
    ids: list[str],
    method: str,
    label_names: list[str],
) -> SecondaryBaseline:
    """Fit a majority or amino-acid lookup baseline from residue labels."""

    rows = _rows_by_id(frame, ids)
    global_counts: Counter[str] = Counter()
    aa_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows.itertuples():
        for aa, label in zip(str(row.sequence), str(row.ss_q3)):
            global_counts[label] += 1
            aa_counts[aa][label] += 1
    global_label = _most_common_label(global_counts, label_names)
    if method == "majority":
        return SecondaryBaseline(method=method, global_label=global_label, aa_to_label={})
    if method == "aa_lookup":
        lookup = {
            aa: _most_common_label(counts, label_names)
            for aa, counts in sorted(aa_counts.items())
        }
        return SecondaryBaseline(method=method, global_label=global_label, aa_to_label=lookup)
    raise ValueError(f"Unsupported secondary-structure baseline method: {method}")


def predict_q3(model: SecondaryBaseline, sequence: str) -> str:
    """Predict a Q3 string with one label per residue."""

    if model.method == "majority":
        return model.global_label * len(sequence)
    if model.method == "aa_lookup":
        return "".join(model.aa_to_label.get(aa, model.global_label) for aa in sequence)
    raise ValueError(f"Unsupported secondary-structure baseline method: {model.method}")


def evaluate_secondary_baseline(
    frame: pd.DataFrame,
    ids: list[str],
    model: SecondaryBaseline,
    label_names: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate a fitted baseline and return metrics plus per-protein predictions."""

    rows = _rows_by_id(frame, ids)
    label_to_index = {label: i for i, label in enumerate(label_names)}
    max_len = int(rows["sequence"].str.len().max()) if len(rows) else 0
    preds = np.zeros((len(rows), max_len), dtype=np.int64)
    targets = np.zeros((len(rows), max_len), dtype=np.int64)
    mask = np.zeros((len(rows), max_len), dtype=bool)
    pred_rows = []
    for i, row in enumerate(rows.itertuples()):
        sequence = str(row.sequence)
        true_q3 = str(row.ss_q3)
        pred_q3 = predict_q3(model, sequence)
        length = len(sequence)
        preds[i, :length] = [label_to_index[label] for label in pred_q3]
        targets[i, :length] = [label_to_index[label] for label in true_q3]
        mask[i, :length] = True
        correct = sum(a == b for a, b in zip(true_q3, pred_q3))
        pred_rows.append(
            {
                "id": str(row.Index),
                "length": length,
                "q3_accuracy": correct / max(length, 1),
                "true_q3": true_q3,
                "pred_q3": pred_q3,
            }
        )
    return compute_secondary_metrics(preds, targets, mask, label_names), pd.DataFrame(pred_rows)
