#!/bin/bash
#SBATCH --job-name=formosan_mt_validate
#SBATCH --account=prudlab
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/scheppat/logs/%x-%j.out
#SBATCH --error=/home/scheppat/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
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

srun --cpu-bind=cores python -u "${EXP_DIR}/scripts/validate_experiment.py" \
  --input "${INPUT}" \
  --target-lang "${TARGET_LANG}" \
  --min-test-ratio "${MIN_TEST_RATIO:-0.075}" \
  --min-validate-ratio "${MIN_VALIDATE_RATIO:-0.025}" \
  --output-json "${OUTPUT_JSON}"
