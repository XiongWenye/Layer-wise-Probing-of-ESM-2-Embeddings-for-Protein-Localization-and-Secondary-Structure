"""Sequence and split validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from esm_probe.constants import VALID_AAS


@dataclass(frozen=True)
class ValidationIssue:
    """A validation issue associated with one sequence."""

    sequence_id: str
    reason: str


def normalize_sequence(sequence: object) -> str:
    """Normalize a raw sequence value."""

    return str(sequence).strip().upper().replace(" ", "").replace("*", "")


def validate_sequence(
    sequence_id: str,
    sequence: str,
    max_length: int,
    truncate_long: bool = False,
    allowed_aas: set[str] | None = None,
) -> tuple[str | None, ValidationIssue | None]:
    """Validate one protein sequence and optionally truncate it."""

    sequence = normalize_sequence(sequence)
    allowed = allowed_aas or VALID_AAS
    if not sequence:
        return None, ValidationIssue(sequence_id, "empty_sequence")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        return None, ValidationIssue(sequence_id, f"invalid_amino_acids:{''.join(invalid)}")
    if len(sequence) > max_length:
        if truncate_long:
            return sequence[:max_length], ValidationIssue(sequence_id, "truncated_long_sequence")
        return None, ValidationIssue(sequence_id, "too_long")
    return sequence, None


def validate_sequence_frame(
    frame: pd.DataFrame,
    max_length: int,
    truncate_long: bool,
    id_col: str = "id",
    seq_col: str = "sequence",
    allowed_aas: set[str] | None = None,
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Validate a frame of sequences and return accepted records plus issues."""

    records: list[dict[str, object]] = []
    issues: list[ValidationIssue] = []
    for row in frame.to_dict(orient="records"):
        seq_id = str(row[id_col])
        sequence, issue = validate_sequence(
            seq_id,
            row[seq_col],
            max_length,
            truncate_long,
            allowed_aas=allowed_aas,
        )
        if issue is not None:
            issues.append(issue)
        if sequence is not None:
            row[seq_col] = sequence
            records.append(row)
    return pd.DataFrame(records), issues


def assert_no_overlap(
    train_ids: Iterable[str],
    external_ids: Iterable[str],
    external_name: str,
) -> None:
    """Raise if training/validation IDs overlap with an external test set."""

    overlap = set(map(str, train_ids)) & set(map(str, external_ids))
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(
            f"Leakage detected: {len(overlap)} IDs overlap with {external_name}: {sample}"
        )


def hash_ids(ids: Iterable[str]) -> str:
    """Hash a collection of IDs for reproducibility manifests."""

    joined = "\n".join(sorted(map(str, ids)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
