"""Small, explicit IO helpers for reproducible artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically to avoid partial artifacts."""

    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: str | Path) -> Any:
    """Read JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, payload: Any) -> None:
    """Write YAML."""

    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV, TSV, parquet, or JSONL table."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV, TSV, or parquet table atomically."""

    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        frame.to_csv(tmp, sep="\t", index=False)
    elif suffix == ".csv":
        frame.to_csv(tmp, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    os.replace(tmp, path)


def find_first_existing(candidates: list[Path]) -> Path | None:
    """Return the first path that exists."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
