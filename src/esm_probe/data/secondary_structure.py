"""CullPDB/CB513 parsing and loading."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from esm_probe.config import Config
from esm_probe.constants import Q3_LABELS, VALID_AAS
from esm_probe.data.validation import assert_no_overlap, validate_sequence_frame
from esm_probe.utils.io import find_first_existing, read_table, write_json, write_table


def _source_path(raw_dir: Path, stem: str) -> Path | None:
    return find_first_existing(
        [
            raw_dir / f"{stem}.csv",
            raw_dir / f"{stem}.tsv",
            raw_dir / f"{stem}.parquet",
            raw_dir / f"{stem}.npy",
            raw_dir / "cullpdb+profile_5926_filtered_updated.npy",
            raw_dir / "cullpdb+profile_5926_filtered.npy",
        ]
    )


AA_ORDER_CULLPDB = list("ACEDGFIHKMLNQPSRTWVYX")
SS8_ORDER_CULLPDB = ["L", "B", "E", "G", "I", "H", "S", "T"]
SS8_TO_Q3 = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "L": "C",
    "S": "C",
    "T": "C",
}


def convert_cullpdb_profile_npy(path: Path) -> pd.DataFrame:
    """Convert the classic CullPDB profile .npy array into id, sequence, ss_q3 rows.

    The public CullPDB profile format has shape N x 700 x 57. Columns [0,22)
    encode residues in the order ACEDGFIHKMLNQPSRTWVYX, with column 21 as
    NoSeq. Columns [22,31) encode DSSP8 labels LBEGIHST plus NoSeq.
    """

    arr = np.load(path, allow_pickle=False, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[1] % (700 * 57) == 0:
        arr = arr.reshape((-1, 700, 57))
    if arr.ndim != 3 or arr.shape[1:] != (700, 57):
        raise ValueError(f"Expected CullPDB profile array with shape N x 700 x 57, got {arr.shape}")

    records: list[dict[str, str]] = []
    for i in range(arr.shape[0]):
        protein = arr[i]
        valid = (protein[:, 21] < 0.5) & (protein[:, 30] < 0.5)
        if not bool(valid.any()):
            continue
        residue_idx = protein[valid, :22].argmax(axis=1)
        ss_idx = protein[valid, 22:31].argmax(axis=1)
        sequence_chars: list[str] = []
        ss_q3_chars: list[str] = []
        for aa_idx, label_idx in zip(residue_idx, ss_idx):
            if aa_idx >= len(AA_ORDER_CULLPDB) or label_idx >= len(SS8_ORDER_CULLPDB):
                continue
            aa = AA_ORDER_CULLPDB[int(aa_idx)]
            ss8 = SS8_ORDER_CULLPDB[int(label_idx)]
            sequence_chars.append(aa)
            ss_q3_chars.append(SS8_TO_Q3[ss8])
        if sequence_chars:
            sequence = "".join(sequence_chars)
            records.append(
                {
                    "id": f"cullpdb_{i:05d}_{hashlib.sha256(sequence.encode('utf-8')).hexdigest()[:10]}",
                    "sequence": sequence,
                    "ss_q3": "".join(ss_q3_chars),
                }
            )
    return pd.DataFrame(records)


def normalize_secondary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize secondary-structure records to id, sequence, ss_q3."""

    lower = {col.lower(): col for col in frame.columns}
    id_col = lower.get("id") or lower.get("protein_id") or lower.get("name")
    seq_col = lower.get("sequence") or lower.get("seq") or lower.get("input")
    ss_col = lower.get("ss_q3") or lower.get("q3") or lower.get("secondary_structure") or lower.get("dssp3")
    if seq_col is None or ss_col is None:
        raise ValueError(
            "Secondary-structure data must include sequence/ss_q3 columns, or input/dssp3 columns."
        )
    out = frame[[seq_col, ss_col]].rename(columns={seq_col: "sequence", ss_col: "ss_q3"})
    if id_col is not None:
        out.insert(0, "id", frame[id_col].astype(str))
    else:
        out.insert(
            0,
            "id",
            [
                f"seq_{i:05d}_{hashlib.sha256(str(seq).encode('utf-8')).hexdigest()[:10]}"
                for i, seq in enumerate(out["sequence"])
            ],
        )
    out["id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].astype(str).str.upper()
    out["ss_q3"] = out["ss_q3"].astype(str).str.upper()
    bad_labels = set("".join(out["ss_q3"].tolist())) - set(Q3_LABELS)
    if bad_labels:
        raise ValueError(f"Unsupported Q3 labels: {sorted(bad_labels)}")
    length_mismatch = out[out["sequence"].str.len() != out["ss_q3"].str.len()]
    if not length_mismatch.empty:
        raise ValueError(f"{len(length_mismatch)} records have sequence/ss_q3 length mismatch.")
    return out


def prepare_secondary_data(config: Config) -> dict[str, object]:
    """Prepare CullPDB/CB513-style secondary-structure tables."""

    raw_dir = Path(config.data.raw_dir)
    processed_dir = Path(config.data.processed_dir)
    train_name = config.data.train_dataset or "cullpdb5926_filtered"
    test_name = config.data.final_test_dataset or "cb513"
    train_path = _source_path(raw_dir, train_name)
    if train_path is None:
        raise FileNotFoundError(f"Could not find raw secondary train table for {train_name}.")
    if train_path.suffix.lower() == ".npy":
        train = convert_cullpdb_profile_npy(train_path)
    else:
        train = normalize_secondary_table(read_table(train_path))
    train, issues = validate_sequence_frame(
        train,
        max_length=config.data.max_sequence_length,
        truncate_long=config.data.truncate_long_sequences,
        allowed_aas=VALID_AAS | {"X"},
    )
    write_table(processed_dir / f"{train_name}.csv", train)

    test_path = _source_path(raw_dir, test_name)
    test_rows = 0
    if test_path is not None:
        test = normalize_secondary_table(read_table(test_path))
        test, test_issues = validate_sequence_frame(
            test,
            max_length=config.data.max_sequence_length,
            truncate_long=config.data.truncate_long_sequences,
            allowed_aas=VALID_AAS | {"X"},
        )
        assert_no_overlap(train["id"], test["id"], test_name)
        write_table(processed_dir / f"{test_name}.csv", test)
        issues.extend(test_issues)
        test_rows = len(test)
    report = {
        "train_dataset": train_name,
        "train_rows": int(len(train)),
        "final_test_dataset": test_name,
        "final_test_rows": int(test_rows),
        "issues": [issue.__dict__ for issue in issues],
    }
    write_json(processed_dir / f"{train_name}_validation_report.json", report)
    return report


def load_secondary_table(config: Config, dataset: str) -> pd.DataFrame:
    """Load a processed secondary-structure table."""

    path = Path(config.data.processed_dir) / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed secondary-structure table not found: {path}")
    return read_table(path)
