"""Artifact path helpers."""

from __future__ import annotations

from pathlib import Path

from esm_probe.config import Config, config_to_dict, runtime_metadata
from esm_probe.utils.io import ensure_dir, write_json, write_yaml


def make_run_dir(config: Config, run_id: str) -> Path:
    """Create and return the run directory."""

    run_dir = Path(config.project.output_dir) / config.project.run_group / run_id
    ensure_dir(run_dir)
    ensure_dir(Path(config.project.output_dir) / "metrics")
    ensure_dir(Path(config.project.output_dir) / "predictions")
    ensure_dir(Path(config.project.output_dir) / "logs")
    ensure_dir(Path(config.project.output_dir) / "figures")
    return run_dir


def save_run_metadata(config: Config, run_dir: Path) -> None:
    """Save resolved config and runtime metadata beside run outputs."""

    write_yaml(run_dir / "config.resolved.yaml", config_to_dict(config))
    write_json(run_dir / "runtime.json", runtime_metadata())
