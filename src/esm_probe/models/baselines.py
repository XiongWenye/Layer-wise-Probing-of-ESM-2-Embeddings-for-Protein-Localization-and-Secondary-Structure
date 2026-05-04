"""Simple sequence baselines for localization."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from esm_probe.constants import STANDARD_AAS


def amino_acid_composition(sequence: str) -> np.ndarray:
    """Compute amino-acid composition features."""

    counts = np.array([sequence.count(aa) for aa in STANDARD_AAS], dtype=np.float32)
    return counts / max(len(sequence), 1)


def dipeptide_composition(sequence: str) -> np.ndarray:
    """Compute dipeptide composition features."""

    pairs = ["".join(pair) for pair in itertools.product(STANDARD_AAS, repeat=2)]
    denom = max(len(sequence) - 1, 1)
    counts = np.array([sequence.count(pair) for pair in pairs], dtype=np.float32)
    return counts / denom


def featurize_sequences(sequences: list[str], method: str) -> np.ndarray:
    """Convert sequences into baseline feature arrays."""

    features: list[np.ndarray] = []
    for sequence in sequences:
        if method == "aac":
            feat = amino_acid_composition(sequence)
        elif method == "dipeptide":
            feat = dipeptide_composition(sequence)
        elif method == "length_aac":
            length = np.array([len(sequence)], dtype=np.float32)
            feat = np.concatenate([length / 1022.0, amino_acid_composition(sequence)])
        else:
            raise ValueError(f"Unsupported baseline feature method: {method}")
        features.append(feat)
    return np.vstack(features)


@dataclass
class _ConstantEstimator:
    """A one-label estimator for folds where only one class is present."""

    constant: int

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = np.zeros((x.shape[0], 2), dtype=np.float32)
        probs[:, self.constant] = 1.0
        return probs


class ConstantAwareOneVsRest:
    """One-vs-rest classifier that tolerates constant labels in a fold."""

    def __init__(self, base_estimator) -> None:
        self.base_estimator = base_estimator
        self.estimators_: list[object] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ConstantAwareOneVsRest":
        self.estimators_ = []
        for j in range(y.shape[1]):
            classes = np.unique(y[:, j])
            if len(classes) == 1:
                self.estimators_.append(_ConstantEstimator(int(classes[0])))
            else:
                estimator = clone(self.base_estimator)
                estimator.fit(x, y[:, j])
                self.estimators_.append(estimator)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        columns = [estimator.predict_proba(x)[:, 1] for estimator in self.estimators_]
        return np.vstack(columns).T


def make_baseline_classifier(class_weight: str | None = "balanced", max_iter: int = 5000):
    """Create a one-vs-rest logistic-regression classifier."""

    base = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        solver="liblinear",
    )
    return ConstantAwareOneVsRest(base)
