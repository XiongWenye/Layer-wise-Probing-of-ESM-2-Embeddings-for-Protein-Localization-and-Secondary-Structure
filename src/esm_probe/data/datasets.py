"""PyTorch datasets for embeddings and labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EmbeddingEntry:
    """One HDF5 embedding entry."""

    sequence_id: str
    dataset_key: str
    length: int


def load_embedding_manifest(path: str | Path) -> dict[str, EmbeddingEntry]:
    """Load an embedding manifest CSV as an ID-indexed dictionary."""

    frame = pd.read_csv(path)
    return {
        str(row.sequence_id): EmbeddingEntry(
            sequence_id=str(row.sequence_id),
            dataset_key=str(row.dataset_key),
            length=int(row.length),
        )
        for row in frame.itertuples(index=False)
    }


class LocalizationEmbeddingDataset(Dataset):
    """Token embedding dataset for sequence-level multilabel localization."""

    def __init__(
        self,
        table: pd.DataFrame,
        label_cols: list[str],
        h5_path: str | Path,
        manifest_path: str | Path,
        ids: list[str],
    ) -> None:
        self.table = table.set_index("id")
        self.label_cols = label_cols
        self.h5_path = Path(h5_path)
        self.manifest = load_embedding_manifest(manifest_path)
        self.ids = [str(seq_id) for seq_id in ids if str(seq_id) in self.manifest]
        missing = sorted(set(map(str, ids)) - set(self.ids))
        if missing:
            raise FileNotFoundError(f"Missing embeddings for {len(missing)} IDs, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        seq_id = self.ids[index]
        entry = self.manifest[seq_id]
        with h5py.File(self.h5_path, "r") as handle:
            emb = np.asarray(handle[entry.dataset_key], dtype=np.float32)
        labels = self.table.loc[seq_id, self.label_cols].to_numpy(dtype=np.float32)
        return torch.from_numpy(emb), torch.from_numpy(labels), seq_id


def collate_token_embeddings(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad variable-length token embeddings."""

    embeddings, labels, ids = zip(*batch)
    lengths = [item.shape[0] for item in embeddings]
    max_len = max(lengths)
    dim = embeddings[0].shape[-1]
    x = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, emb in enumerate(embeddings):
        x[i, : emb.shape[0]] = emb
        mask[i, : emb.shape[0]] = True
    y = torch.stack(labels)
    return x, mask, y, list(ids)


class SecondaryEmbeddingDataset(Dataset):
    """Token embedding dataset for residue-level secondary-structure labels."""

    def __init__(
        self,
        table: pd.DataFrame,
        h5_path: str | Path,
        manifest_path: str | Path,
        ids: list[str],
        label_to_index: dict[str, int],
    ) -> None:
        self.table = table.set_index("id")
        self.h5_path = Path(h5_path)
        self.manifest = load_embedding_manifest(manifest_path)
        self.ids = [str(seq_id) for seq_id in ids if str(seq_id) in self.manifest]
        self.label_to_index = label_to_index
        missing = sorted(set(map(str, ids)) - set(self.ids))
        if missing:
            raise FileNotFoundError(f"Missing embeddings for {len(missing)} IDs, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        seq_id = self.ids[index]
        entry = self.manifest[seq_id]
        with h5py.File(self.h5_path, "r") as handle:
            emb = np.asarray(handle[entry.dataset_key], dtype=np.float32)
        labels = np.array([self.label_to_index[ch] for ch in self.table.loc[seq_id, "ss_q3"]], dtype=np.int64)
        if len(labels) != emb.shape[0]:
            raise ValueError(f"Embedding/label length mismatch for {seq_id}")
        return torch.from_numpy(emb), torch.from_numpy(labels), seq_id


def collate_secondary_embeddings(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad variable-length residue embeddings and labels."""

    embeddings, labels, ids = zip(*batch)
    lengths = [item.shape[0] for item in embeddings]
    max_len = max(lengths)
    dim = embeddings[0].shape[-1]
    x = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    y = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (emb, lab) in enumerate(zip(embeddings, labels)):
        x[i, : emb.shape[0]] = emb
        y[i, : lab.shape[0]] = lab
        mask[i, : emb.shape[0]] = True
    return x, mask, y, list(ids)
