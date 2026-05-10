#!/bin/bash
#SBATCH --job-name=formosan_mt_splits
#SBATCH --account=prudlab
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/scheppat/logs/%x-%j.out
#SBATCH --error=/home/scheppat/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

EXP_DIR="${EXP_DIR:-/home/scheppat/formosan_mt_experiments}"
INPUT="${INPUT:-/projects/prudlab/formosan_parallel_corpora/big_corpus_en.csv}"
OUT_DIR="${OUT_DIR:-/scratch/scheppat/formosan_mt_experiments/data/splits_en_v1}"

mkdir -p "${OUT_DIR}"

python -u "${EXP_DIR}/scripts/build_experiment_splits.py" \
  --input "${INPUT}" \
  --output-dir "${OUT_DIR}" \
  --train-ratio 0.90 \
  --val-ratio 0.025 \
  --test-ratio 0.075 \
  --tiers lexical,in_domain_hard,hard_global
