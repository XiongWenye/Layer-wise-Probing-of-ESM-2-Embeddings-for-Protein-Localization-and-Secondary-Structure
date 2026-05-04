"""Project-wide constants."""

from __future__ import annotations

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
VALID_AAS = set(STANDARD_AAS)

LOCALIZATION_LABELS = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]

Q3_LABELS = ["H", "E", "C"]

ESM_LAYER_COUNTS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
    "esm2_t48_15B_UR50D": 48,
}

ESM_EMBED_DIMS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t48_15B_UR50D": 5120,
}
