import numpy as np

from esm_probe.evaluation.localization_metrics import compute_localization_metrics, tune_multilabel_thresholds
from esm_probe.evaluation.secondary_metrics import compute_secondary_metrics


def test_localization_metrics_perfect() -> None:
    y = np.array([[1, 0], [0, 1], [1, 1]])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.9, 0.8]])
    t = tune_multilabel_thresholds(p, y)
    metrics = compute_localization_metrics(p, y, t, ["a", "b"])
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_secondary_metrics_masked() -> None:
    pred = np.array([[0, 1, 2], [0, 0, 0]])
    target = np.array([[0, 1, 1], [0, 1, 2]])
    mask = np.array([[1, 1, 1], [1, 0, 0]])
    metrics = compute_secondary_metrics(pred, target, mask, ["H", "E", "C"])
    assert metrics["q3_accuracy"] == 0.75
