#!/bin/bash
#SBATCH --job-name=formosan_mt_bootstrap
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXP_DIR="${EXP_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
: "${PREDICTIONS:?Set PREDICTIONS to predictions.csv}"
: "${METRICS:?Set METRICS to metrics.json}"

srun --cpu-bind=cores \
  python -u "${EXP_DIR}/scripts/bootstrap_predictions.py" \
  --predictions "${PREDICTIONS}" \
  --metrics "${METRICS}" \
  --samples "${BOOTSTRAP_SAMPLES:-200}" \
  --seed "${BOOTSTRAP_SEED:-42}" \
  --workers "${BOOTSTRAP_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
