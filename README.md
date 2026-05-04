# ESM-2 Layer Probing for Protein Localization and Secondary Structure

This repository supports the MCB128 final project:

**Layer-wise probing of ESM-2 embeddings for protein subcellular localization and secondary structure**

The codebase is config-driven and designed to preserve official data splits:

- DeepLoc 2.0 five-fold homology-aware cross-validation
- HPA held out until final external localization evaluation
- CullPDB5926-filtered train/validation for secondary structure
- CB513 held out until final secondary-structure evaluation

## Setup

```bash
mamba env create -f environment.yml
conda activate mcb128-esm-probe
pip install -e .
```

## Expected Data Layout

Place official or pre-downloaded source files under `data/raw/`. The preparation script accepts flexible CSV/TSV/FASTA-style inputs, but recommended normalized raw names are:

```text
data/raw/deeploc2.csv
data/raw/hpa.csv
data/raw/deeploc2_folds.csv
data/raw/cullpdb5926_filtered.csv
data/raw/cb513.csv
```

Localization tables should contain an ID column, a sequence column, and either multilabel columns or a label/list column. Split tables should contain `id` and `fold`.

## Milestone 1 Commands

```bash
python scripts/download_or_prepare_data.py --config configs/deeploc_baselines.yaml
python scripts/train_localization_baseline.py --config configs/deeploc_baselines.yaml --override 'data.folds=[0]' 'baseline.methods=[aac]'
python scripts/extract_esm_embeddings.py --config configs/deeploc_esm_layers.yaml --override 'model.esm_names=[esm2_t12_35M_UR50D]' 'model.layers=[6,12]'
python scripts/train_localization_probe.py --config configs/deeploc_esm_layers.yaml --override 'data.folds=[0]' 'training.seeds=[1]' 'probe.pooling=[mean]' 'probe.hidden_dim=[128]' 'probe.dropout=[0.1]' 'training.learning_rate=[0.001]' 'training.weight_decay=[0.0001]'
```

Each run writes resolved configs, metrics, predictions, logs, and checkpoints under `results/`.

## Final Evaluation

Only run HPA or CB513 evaluation after selecting models using validation/CV results:

```bash
python scripts/summarize_results.py --config configs/deeploc_esm_layers.yaml
python scripts/evaluate_localization.py \
  --config configs/deeploc_esm_layers.yaml \
  --selected-run-id <selected_deeploc_run_id>
python scripts/evaluate_secondary_structure.py \
  --config configs/cb513_probe.yaml \
  --selected-run-id <selected_secondary_run_id>
python scripts/make_figures.py --config configs/deeploc_esm_layers.yaml
```

## SLURM

```bash
sbatch slurm/esm_embed.sbatch scripts/extract_esm_embeddings.py --config configs/deeploc_esm_layers.yaml
sbatch slurm/train_probe.sbatch scripts/train_localization_probe.py --config configs/deeploc_esm_layers.yaml
```

## Reproducibility

Every run saves the resolved config, command line, host, Python version, package versions where available, CUDA availability, split hashes, metrics, predictions, and logs. The code rejects unsupported ESM layers by default and performs split-leakage checks before training.


For the current paper-ready scope, use frozen ESM-2 35M embeddings, layers 6 and 12,
mean/max/attention pooling for localization, classical sequence baselines, and Q3
secondary structure as a supporting extension.
