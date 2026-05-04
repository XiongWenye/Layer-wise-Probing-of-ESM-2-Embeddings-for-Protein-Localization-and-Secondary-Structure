"""Report aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from esm_probe.utils.io import read_json


def load_metric_files(metrics_dir: str | Path) -> pd.DataFrame:
    """Load metric JSON files into a flat table where possible."""

    rows: list[dict[str, object]] = []
    for path in Path(metrics_dir).glob("*.json"):
        payload = read_json(path)
        row = {"metric_file": str(path), **payload}
        for key in ["per_label_mcc", "auroc_per_label", "thresholds"]:
            row.pop(key, None)
        rows.append(row)
    return pd.DataFrame(rows)
