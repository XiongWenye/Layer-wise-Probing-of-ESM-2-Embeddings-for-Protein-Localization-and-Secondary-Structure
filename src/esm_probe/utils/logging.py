"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from esm_probe.utils.io import ensure_dir


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and optional file logging."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
