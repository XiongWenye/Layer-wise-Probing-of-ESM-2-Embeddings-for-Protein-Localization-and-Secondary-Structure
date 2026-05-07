#!/bin/bash
#SBATCH --job-name=esm-embed
#SBATCH --output=results/logs/%j.out
#SBATCH --error=results/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --account=
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate mcb128-esm-probe
set -u

python "$@"
