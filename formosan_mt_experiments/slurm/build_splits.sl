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

EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
TARGET_LANG="${TARGET_LANG:-english}"
case "${TARGET_LANG}" in
  english)
    TARGET_COL="${TARGET_COL:-english_sentence}"
    FILE_SHORT="en"
    ;;
  chinese)
    TARGET_COL="${TARGET_COL:-chinese_sentence}"
    FILE_SHORT="zh"
    ;;
  *)
    echo "Unsupported TARGET_LANG=${TARGET_LANG}" >&2
    exit 1
    ;;
esac
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
if [[ -n "${CORPUS_NAME:-}" ]]; then
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}/${CORPUS_NAME}}"
else
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}}"
fi
INPUT="${INPUT:-${CORPUS_DIR}/big_corpus_${FILE_SHORT}.csv}"
OUT_DIR="${OUT_DIR:-${CORPUS_DIR}/splits_${FILE_SHORT}_v1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-big_corpus_${FILE_SHORT}}"

mkdir -p "${OUT_DIR}"

python -u "${EXP_DIR}/scripts/build_experiment_splits.py" \
  --input "${INPUT}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --output-dir "${OUT_DIR}" \
  --train-ratio 0.90 \
  --val-ratio 0.025 \
  --test-ratio 0.075 \
  --min-formosan-tokens "${MIN_FORMOSAN_TOKENS:-4}" \
  --min-target-tokens "${MIN_TARGET_TOKENS:-4}" \
  --tiers lexical,in_domain_hard,hard_global
