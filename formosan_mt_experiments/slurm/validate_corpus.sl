#!/bin/bash
#SBATCH --job-name=formosan_mt_validate
#SBATCH --partition=short
#SBATCH --time=02:00:00
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

EXP_DIR="${EXP_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_DATA="${PROJECT_DATA:-${EXP_DIR}/data/corpora}"
TIER="${TIER:-in_domain_hard}"
TARGET_LANG="${TARGET_LANG:?TARGET_LANG is required}"
CORPUS_NAME="${CORPUS_NAME:?CORPUS_NAME is required}"

case "${TARGET_LANG}" in
  english) SHORT=en ;;
  chinese) SHORT=zh ;;
  *) echo "Unsupported TARGET_LANG=${TARGET_LANG}" >&2; exit 1 ;;
esac

INPUT="${INPUT:-${PROJECT_DATA}/${CORPUS_NAME}/big_corpus_${SHORT}_${TIER}.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-${PROJECT_DATA}/${CORPUS_NAME}/provenance/validate_${SHORT}_${TIER}_runtime.json}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/default_experiment.json}"

srun --cpu-bind=cores python -u "${EXP_DIR}/scripts/validate_experiment.py" \
  --input "${INPUT}" \
  --profile "${PROFILE}" \
  --target-lang "${TARGET_LANG}" \
  --output-json "${OUTPUT_JSON}"
