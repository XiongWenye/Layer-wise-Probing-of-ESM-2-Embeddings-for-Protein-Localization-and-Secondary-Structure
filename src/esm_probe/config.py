"""Typed configuration loading and command-line override support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    """Run-level project settings."""

    model_config = ConfigDict(extra="allow")

    name: str = "esm_layer_probe"
    run_group: str = "default"
    output_dir: Path = Path("results")
    cache_dir: Path = Path("data/processed/cache")
    seed: int = 1
    deterministic: bool = True
    num_workers: int = 4
    device: Literal["auto", "cpu", "cuda"] = "auto"


class DataConfig(BaseModel):
    """Dataset and split settings."""

    model_config = ConfigDict(extra="allow")

    task: Literal["localization", "secondary_structure"] = "localization"
    dataset: str | None = None
    external_test: str | None = None
    train_dataset: str | None = None
    final_test_dataset: str | None = None
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    splits_dir: Path = Path("data/splits")
    folds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    preserve_official_splits: bool = True
    max_sequence_length: int = 1022
    truncate_long_sequences: bool = False
    validate_sequences: bool = True


class ModelConfig(BaseModel):
    """ESM model and embedding settings."""

    model_config = ConfigDict(extra="allow")

    esm_name: str = "esm2_t12_35M_UR50D"
    esm_names: list[str] = Field(default_factory=lambda: ["esm2_t12_35M_UR50D"])
    layers: list[int] = Field(default_factory=lambda: [6, 12])
    allow_invalid_layers: Literal["error", "skip"] = "error"
    freeze_esm: bool = True
    embedding_storage: str = "hdf5"


class BaselineConfig(BaseModel):
    """Classical sequence baseline settings."""

    model_config = ConfigDict(extra="allow")

    methods: list[str] = Field(default_factory=lambda: ["aac"])
    classifier: str = "logistic_regression"
    class_weight: str | None = "balanced"
    max_iter: int = 5000


class ProbeConfig(BaseModel):
    """Neural probe hyperparameters."""

    model_config = ConfigDict(extra="allow")

    pooling: list[str] = Field(default_factory=lambda: ["mean"])
    hidden_dim: list[int] = Field(default_factory=lambda: [128])
    dropout: list[float] = Field(default_factory=lambda: [0.1])
    activation: str = "gelu"
    output: str = "multilabel_sigmoid"
    architecture: str | None = None


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    model_config = ConfigDict(extra="allow")

    batch_size: int = 32
    epochs: int = 20
    patience: int = 5
    learning_rate: float | list[float] = 1e-3
    weight_decay: float | list[float] = 1e-4
    seeds: list[int] = Field(default_factory=lambda: [1])
    loss: str = "bce_with_logits"
    threshold_selection: str = "validation_macro_f1"
    amp: bool = True
    gradient_clip_norm: float = 1.0


class LoggingConfig(BaseModel):
    """Logging and artifact settings."""

    model_config = ConfigDict(extra="allow")

    level: str = "INFO"
    save_predictions: bool = True
    save_train_curves: bool = True


class MetricsConfig(BaseModel):
    """Metric selection."""

    model_config = ConfigDict(extra="allow")

    localization: list[str] = Field(default_factory=list)
    secondary_structure: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="allow")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def parse_args(description: str | None = None) -> argparse.Namespace:
    """Parse the common script interface."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--selected-run-id",
        default=None,
        help="Run ID to use for deterministic final evaluation scripts.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Overrides as dotted.path=value, parsed as YAML values.",
    )
    return parser.parse_args()


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override through non-mapping key: {dotted_key}")
        cursor = child
    cursor[parts[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides to a raw config dictionary."""

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have form key=value, got: {override}")
        key, value_text = override.split("=", 1)
        value = yaml.safe_load(value_text)
        _set_nested(raw, key, value)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load and validate a YAML config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = apply_overrides(raw, overrides or [])
    return Config.model_validate(raw)


def config_to_dict(config: Config) -> dict[str, Any]:
    """Return a JSON/YAML-serializable config dictionary."""

    return json.loads(config.model_dump_json())


def stable_hash(payload: Any, length: int = 12) -> str:
    """Compute a short stable hash for run IDs and manifests."""

    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def make_run_id(prefix: str, payload: Any) -> str:
    """Build a deterministic run ID from a prefix and payload."""

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in prefix).strip("-")
    return f"{safe}-{stable_hash(payload)}"


def get_git_commit() -> str | None:
    """Return the current git commit if this directory is a git repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def runtime_metadata() -> dict[str, Any]:
    """Collect lightweight reproducibility metadata."""

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        torch_version = torch.__version__
    except Exception:
        cuda_available = False
        torch_version = None

    return {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": get_git_commit(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
    }
