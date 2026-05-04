"""DeepLoc/HPA parsing, preparation, and split loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import pandas as pd

from esm_probe.config import Config
from esm_probe.constants import LOCALIZATION_LABELS
from esm_probe.data.validation import assert_no_overlap, hash_ids, validate_sequence_frame
from esm_probe.utils.io import find_first_existing, read_table, write_json, write_table

LOGGER = logging.getLogger(__name__)

ID_CANDIDATES = ["id", "protein_id", "entry", "accession", "acc", "sid", "uniprot", "name"]
SEQ_CANDIDATES = ["sequence", "seq", "protein_sequence", "fasta"]
LABEL_CANDIDATES = ["labels", "label", "localizations", "localization", "target", "targets"]
FOLD_CANDIDATES = ["fold", "cv_fold", "split_fold", "partition"]

DEEPlOC_OFFICIAL_URLS = {
    "deeploc2": "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv",
    "hpa": "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/hpa_testset.csv",
    "sorting_signals": "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/SortingSignalsSwissprot.csv",
}


def _find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower_to_actual = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _read_fasta(path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    current_id: str | None = None
    current_seq: list[str] = []
    current_labels: str = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append(
                    {"id": current_id, "sequence": "".join(current_seq), "labels": current_labels}
                )
            header = line[1:]
            parts = header.replace("|", " ").split()
            current_id = parts[0]
            current_labels = ";".join(parts[1:])
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        records.append({"id": current_id, "sequence": "".join(current_seq), "labels": current_labels})
    return pd.DataFrame(records)


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".fa", ".faa", ".fasta"}:
        return _read_fasta(path)
    return read_table(path)


def _source_path(raw_dir: Path, stem: str) -> Path | None:
    candidates = [
        raw_dir / f"{stem}.csv",
        raw_dir / f"{stem}.tsv",
        raw_dir / f"{stem}.parquet",
        raw_dir / f"{stem}.fasta",
        raw_dir / f"{stem}.fa",
    ]
    return find_first_existing(candidates)


def _source_or_download(raw_dir: Path, stem: str) -> Path | None:
    """Return a local raw source path, downloading known public DeepLoc files if absent."""

    existing = _source_path(raw_dir, stem)
    if existing is not None:
        return existing
    url = DEEPlOC_OFFICIAL_URLS.get(stem)
    if url is None:
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{stem}.csv"
    LOGGER.info("Downloading %s to %s", url, destination)
    urlretrieve(url, destination)
    return destination


def _split_labels(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value)
    for sep in ["|", ",", ";"]:
        text = text.replace(sep, ";")
    return {part.strip() for part in text.split(";") if part.strip()}


def normalize_localization_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw localization table to id, sequence, labels, and binary label columns."""

    id_col = _find_col(frame.columns, ID_CANDIDATES)
    seq_col = _find_col(frame.columns, SEQ_CANDIDATES)
    if id_col is None or seq_col is None:
        raise ValueError("Localization data must include an ID column and a sequence column.")

    normalized = pd.DataFrame(
        {
            "id": frame[id_col].astype(str),
            "sequence": frame[seq_col].astype(str),
        }
    )

    binary_cols = [col for col in frame.columns if col in LOCALIZATION_LABELS]
    label_col = _find_col(frame.columns, LABEL_CANDIDATES)
    labels_per_row: list[set[str]] = []
    if binary_cols:
        for _, row in frame.iterrows():
            labels_per_row.append({label for label in binary_cols if int(row[label]) == 1})
        all_labels = LOCALIZATION_LABELS
    elif label_col is not None:
        labels_per_row = [_split_labels(value) for value in frame[label_col]]
        all_labels = sorted({label for labels in labels_per_row for label in labels})
    else:
        excluded = {id_col, seq_col, *FOLD_CANDIDATES}
        candidate_cols = [col for col in frame.columns if col not in excluded]
        if not candidate_cols:
            raise ValueError("No localization labels found.")
        for _, row in frame.iterrows():
            labels_per_row.append({str(col) for col in candidate_cols if int(row[col]) == 1})
        all_labels = sorted({label for labels in labels_per_row for label in labels})

    if not all_labels:
        raise ValueError("All localization records have empty labels.")
    normalized["labels"] = [";".join(sorted(labels)) for labels in labels_per_row]
    for label in all_labels:
        normalized[label] = [int(label in labels) for labels in labels_per_row]

    fold_col = _find_col(frame.columns, FOLD_CANDIDATES)
    if fold_col is not None:
        normalized["fold"] = frame[fold_col].astype(int)
    return normalized


def prepare_localization_data(config: Config) -> dict[str, object]:
    """Prepare DeepLoc and optional HPA tables plus split manifests."""

    raw_dir = Path(config.data.raw_dir)
    processed_dir = Path(config.data.processed_dir)
    splits_dir = Path(config.data.splits_dir)
    dataset = config.data.dataset or "deeploc2"
    external = config.data.external_test or "hpa"

    source = _source_or_download(raw_dir, dataset)
    if source is None:
        raise FileNotFoundError(
            f"Could not find raw {dataset} table in {raw_dir}. Expected {dataset}.csv/tsv/parquet/fasta."
        )
    LOGGER.info("Preparing localization data from %s", source)
    table = normalize_localization_table(_read_raw(source))
    table, issues = validate_sequence_frame(
        table,
        max_length=config.data.max_sequence_length,
        truncate_long=config.data.truncate_long_sequences,
    )
    write_table(processed_dir / f"{dataset}.csv", table)

    fold_path = _source_path(raw_dir, f"{dataset}_folds")
    if fold_path is not None:
        fold_table = read_table(fold_path)
        id_col = _find_col(fold_table.columns, ID_CANDIDATES) or "id"
        fold_col = _find_col(fold_table.columns, FOLD_CANDIDATES) or "fold"
        splits = fold_table[[id_col, fold_col]].rename(columns={id_col: "id", fold_col: "fold"})
    elif "fold" in table.columns:
        splits = table[["id", "fold"]].copy()
    elif config.data.preserve_official_splits:
        raise FileNotFoundError(
            "Official DeepLoc folds are required. Provide data/raw/deeploc2_folds.csv "
            "or a fold column in data/raw/deeploc2.csv."
        )
    else:
        raise ValueError("Random split creation is intentionally not implemented to avoid leakage.")
    splits["id"] = splits["id"].astype(str)
    splits["fold"] = splits["fold"].astype(int)
    missing_ids = set(splits["id"]) - set(table["id"])
    if missing_ids:
        raise ValueError(f"Split manifest contains IDs absent from processed data: {sorted(missing_ids)[:10]}")
    write_table(splits_dir / f"{dataset}_folds.csv", splits)

    external_path = _source_or_download(raw_dir, external)
    external_rows = 0
    if external_path is not None:
        ext_table = normalize_localization_table(_read_raw(external_path))
        ext_table, ext_issues = validate_sequence_frame(
            ext_table,
            max_length=config.data.max_sequence_length,
            truncate_long=config.data.truncate_long_sequences,
        )
        assert_no_overlap(table["id"], ext_table["id"], external)
        write_table(processed_dir / f"{external}.csv", ext_table)
        issues.extend(ext_issues)
        external_rows = len(ext_table)
    else:
        LOGGER.warning("No external localization set found for %s; final evaluation will require it.", external)

    report = {
        "dataset": dataset,
        "rows": int(len(table)),
        "labels": [col for col in table.columns if col not in {"id", "sequence", "labels", "fold"}],
        "folds": sorted(splits["fold"].unique().tolist()),
        "id_hash": hash_ids(table["id"]),
        "external_test": external,
        "external_rows": external_rows,
        "issues": [issue.__dict__ for issue in issues],
    }
    write_json(processed_dir / f"{dataset}_validation_report.json", report)
    return report


def load_localization_table(config: Config, dataset: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load a processed localization table and return table plus label columns."""

    dataset = dataset or config.data.dataset or "deeploc2"
    path = Path(config.data.processed_dir) / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed localization table not found: {path}")
    table = read_table(path)
    ignored = {"id", "sequence", "labels", "fold"}
    label_cols = [col for col in table.columns if col not in ignored]
    if not label_cols:
        raise ValueError(f"No label columns found in {path}")
    return table, label_cols


def load_fold_split(config: Config, fold: int) -> tuple[list[str], list[str]]:
    """Load train/validation IDs for one official fold."""

    dataset = config.data.dataset or "deeploc2"
    path = Path(config.data.splits_dir) / f"{dataset}_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    splits = read_table(path)
    splits["id"] = splits["id"].astype(str)
    train_ids = splits.loc[splits["fold"].astype(int) != fold, "id"].tolist()
    val_ids = splits.loc[splits["fold"].astype(int) == fold, "id"].tolist()
    if not train_ids or not val_ids:
        raise ValueError(f"Fold {fold} has empty train or validation IDs.")
    return train_ids, val_ids
