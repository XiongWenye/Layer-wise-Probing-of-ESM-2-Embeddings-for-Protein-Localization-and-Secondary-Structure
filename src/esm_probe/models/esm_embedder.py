"""ESM-2 loading, layer validation, and HDF5 embedding writes."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import h5py
import pandas as pd
import torch
from tqdm import tqdm

from esm_probe.constants import ESM_LAYER_COUNTS
from esm_probe.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve CPU/GPU execution."""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_layers(esm_name: str, layers: list[int], allow_invalid: str = "error") -> list[int]:
    """Validate requested ESM representation layers for a model."""

    max_layer = ESM_LAYER_COUNTS.get(esm_name)
    if max_layer is None:
        LOGGER.warning("Unknown ESM model %s; layer validation will rely on runtime model.", esm_name)
        return layers
    valid = [layer for layer in layers if 0 <= int(layer) <= max_layer]
    invalid = sorted(set(layers) - set(valid))
    if invalid and allow_invalid == "error":
        raise ValueError(f"{esm_name} supports layers 0..{max_layer}; invalid requested layers: {invalid}")
    if invalid:
        LOGGER.warning("Skipping invalid layers for %s: %s", esm_name, invalid)
    return valid


def load_esm_model(esm_name: str, device: torch.device):
    """Load an ESM-2 model by name from fair-esm."""

    try:
        import esm
    except ImportError as exc:
        raise ImportError("fair-esm is required for embedding extraction. Install with `pip install fair-esm`.") from exc
    if not hasattr(esm.pretrained, esm_name):
        raise ValueError(f"fair-esm does not expose pretrained model {esm_name}")
    model, alphabet = getattr(esm.pretrained, esm_name)()
    model.eval()
    model.to(device)
    return model, alphabet


def embedding_paths(processed_dir: str | Path, dataset: str, esm_name: str, layer: int) -> tuple[Path, Path]:
    """Return HDF5 and manifest paths for one layer."""

    base = Path(processed_dir) / "embeddings" / dataset / esm_name
    return base / f"layer_{layer}.h5", base / f"layer_{layer}_manifest.csv"


def _dataset_key(sequence_id: str) -> str:
    digest = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:16]
    return f"embeddings/{digest}"


def sequence_hash(sequence: str) -> str:
    """Hash a protein sequence."""

    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def extract_embeddings_to_hdf5(
    table: pd.DataFrame,
    dataset: str,
    esm_name: str,
    layers: list[int],
    processed_dir: str | Path,
    batch_size: int,
    device_name: str,
    allow_invalid_layers: str,
    amp: bool,
) -> list[Path]:
    """Extract per-token ESM embeddings to HDF5 layer files."""

    device = resolve_device(device_name)
    layers = validate_layers(esm_name, layers, allow_invalid_layers)
    if not layers:
        raise ValueError(f"No valid layers remain for {esm_name}.")
    model, alphabet = load_esm_model(esm_name, device)
    batch_converter = alphabet.get_batch_converter()
    output_paths: list[Path] = []

    for layer in layers:
        h5_path, manifest_path = embedding_paths(processed_dir, dataset, esm_name, layer)
        ensure_dir(h5_path.parent)
        if h5_path.exists() and manifest_path.exists():
            LOGGER.info("Embedding cache exists for %s layer %s; skipping.", esm_name, layer)
            output_paths.append(h5_path)
            continue
        rows: list[dict[str, object]] = []
        with h5py.File(h5_path, "w") as handle:
            for start in tqdm(range(0, len(table), batch_size), desc=f"{esm_name} layer {layer}"):
                chunk = table.iloc[start : start + batch_size]
                batch = [(str(row.id), str(row.sequence)) for row in chunk.itertuples(index=False)]
                _, _, toks = batch_converter(batch)
                toks = toks.to(device)
                with torch.no_grad():
                    use_amp = amp and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        reps = model(toks, repr_layers=[layer], return_contacts=False)["representations"][layer]
                for i, (seq_id, sequence) in enumerate(batch):
                    length = len(sequence)
                    token_rep = reps[i, 1 : length + 1].detach().cpu().float().numpy()
                    key = _dataset_key(seq_id)
                    handle.create_dataset(key, data=token_rep, compression="gzip")
                    handle[key].attrs["sequence_id"] = seq_id
                    handle[key].attrs["length"] = length
                    rows.append(
                        {
                            "sequence_id": seq_id,
                            "dataset_key": key,
                            "length": length,
                            "sequence_sha256": sequence_hash(sequence),
                            "esm_name": esm_name,
                            "layer": layer,
                        }
                    )
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        output_paths.append(h5_path)
    return output_paths
