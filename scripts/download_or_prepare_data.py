#!/usr/bin/env python
"""Prepare raw data into normalized processed tables and official split manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import parse_args, load_config
from esm_probe.data.deeploc import prepare_localization_data
from esm_probe.data.secondary_structure import prepare_secondary_data
from esm_probe.utils.logging import setup_logging


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    if cfg.data.task == "localization":
        report = prepare_localization_data(cfg)
    else:
        report = prepare_secondary_data(cfg)
    print(report)


if __name__ == "__main__":
    main()
