"""Multilabel localization metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, matthews_corrcoef, roc_auc_score


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    return 1.0 / (1.0 + np.exp(-logits))


def tune_multilabel_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    metric: str = "macro_f1",
) -> np.ndarray:
    """Tune one threshold per label on validation data."""

    del metric
    thresholds = np.full(probabilities.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(probabilities.shape[1]):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in grid:
            pred = (probabilities[:, j] >= threshold).astype(int)
            score = f1_score(targets[:, j], pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[j] = best_threshold
    return thresholds


def compute_localization_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None,
    label_names: list[str],
) -> dict[str, object]:
    """Compute multilabel localization metrics."""

    if thresholds is None:
        thresholds = np.full(probabilities.shape[1], 0.5)
    preds = (probabilities >= thresholds.reshape(1, -1)).astype(int)
    metrics: dict[str, object] = {
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "jaccard": float(jaccard_score(targets, preds, average="samples", zero_division=0)),
        "thresholds": {label: float(thresholds[i]) for i, label in enumerate(label_names)},
    }
    per_label_mcc: dict[str, float] = {}
    auroc_per_label: dict[str, float | None] = {}
    for i, label in enumerate(label_names):
        per_label_mcc[label] = float(matthews_corrcoef(targets[:, i], preds[:, i]))
        try:
            auroc_per_label[label] = float(roc_auc_score(targets[:, i], probabilities[:, i]))
        except ValueError:
            auroc_per_label[label] = None
    metrics["per_label_mcc"] = per_label_mcc
    metrics["auroc_per_label"] = auroc_per_label
    return metrics
