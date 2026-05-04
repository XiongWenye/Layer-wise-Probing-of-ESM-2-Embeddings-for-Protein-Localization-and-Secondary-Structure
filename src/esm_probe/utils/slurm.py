"""SLURM script helpers."""

from __future__ import annotations


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=results/logs/%j.out
#SBATCH --error=results/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_ydu_lab
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G

set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate mcb128-esm-probe

python "$@"
"""


def render_slurm(job_name: str) -> str:
    """Render the standard single-GPU SLURM wrapper."""

    return SLURM_TEMPLATE.format(job_name=job_name)
