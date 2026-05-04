"""Secondary-structure metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def compute_secondary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    """Compute masked Q3 accuracy and per-class F1."""

    valid_pred = predictions[mask.astype(bool)]
    valid_targets = targets[mask.astype(bool)]
    q3 = float((valid_pred == valid_targets).mean()) if len(valid_targets) else 0.0
    per_class = f1_score(
        valid_targets,
        valid_pred,
        labels=list(range(len(label_names))),
        average=None,
        zero_division=0,
    )
    return {
        "q3_accuracy": q3,
        "per_class_f1": {label: float(per_class[i]) for i, label in enumerate(label_names)},
    }
