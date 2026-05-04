"""Training-curve loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_training_curves(results_dir: str | Path) -> pd.DataFrame:
    """Load all training curve CSV files under a results directory."""

    rows: list[pd.DataFrame] = []
    for path in Path(results_dir).glob("**/training_curve.csv"):
        frame = pd.read_csv(path)
        frame["run_id"] = path.parent.name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
