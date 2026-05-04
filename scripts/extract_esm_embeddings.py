#!/usr/bin/env python
"""Extract frozen ESM-2 token embeddings to HDF5."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esm_probe.config import load_config, parse_args
from esm_probe.data.deeploc import load_localization_table
from esm_probe.data.secondary_structure import load_secondary_table
from esm_probe.models.esm_embedder import extract_embeddings_to_hdf5
from esm_probe.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def _model_names(cfg) -> list[str]:
    return cfg.model.esm_names or [cfg.model.esm_name]


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.override)
    setup_logging(cfg.logging.level)
    datasets = []
    if cfg.data.task == "localization":
        dataset = cfg.data.dataset or "deeploc2"
        table, _ = load_localization_table(cfg, dataset)
        datasets.append((dataset, table))
        if cfg.data.external_test:
            try:
                ext_table, _ = load_localization_table(cfg, cfg.data.external_test)
                datasets.append((cfg.data.external_test, ext_table))
            except FileNotFoundError:
                LOGGER.warning("External set %s is not processed yet; skipping embeddings.", cfg.data.external_test)
    else:
        dataset = cfg.data.train_dataset or "cullpdb5926_filtered"
        table = load_secondary_table(cfg, dataset)
        datasets.append((dataset, table))
        if cfg.data.final_test_dataset:
            try:
                test_table = load_secondary_table(cfg, cfg.data.final_test_dataset)
                datasets.append((cfg.data.final_test_dataset, test_table))
            except FileNotFoundError:
                LOGGER.warning("Final test set %s is not processed yet; skipping embeddings.", cfg.data.final_test_dataset)
    for esm_name in _model_names(cfg):
        for dataset_name, dataset_table in datasets:
            LOGGER.info("Extracting %s for %s", esm_name, dataset_name)
            extract_embeddings_to_hdf5(
                table=dataset_table,
                dataset=dataset_name,
                esm_name=esm_name,
                layers=cfg.model.layers,
                processed_dir=cfg.data.processed_dir,
                batch_size=cfg.training.batch_size,
                device_name=cfg.project.device,
                allow_invalid_layers=cfg.model.allow_invalid_layers,
                amp=cfg.training.amp,
            )


if __name__ == "__main__":
    main()
