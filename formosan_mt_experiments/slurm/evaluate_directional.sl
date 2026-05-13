#!/bin/bash
#SBATCH --job-name=formosan_mt_eval
#SBATCH --account=prudlab
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vr80g|vr144g
#SBATCH --output=/home/scheppat/logs/%x-%j.out
#SBATCH --error=/home/scheppat/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

export PYTHONUNBUFFERED=1

EXP_DIR="${EXP_DIR:-/home/scheppat/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat}"
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
TIER="${TIER:-in_domain_hard}"
TARGET_LANG="${TARGET_LANG:-english}"
case "${TARGET_LANG}" in
  english)
    TARGET_COL="${TARGET_COL:-english_sentence}"
    FILE_SHORT="en"
    DEFAULT_DIRECTION="f2en"
    ;;
  chinese)
    TARGET_COL="${TARGET_COL:-chinese_sentence}"
    FILE_SHORT="zh"
    DEFAULT_DIRECTION="f2zh"
    ;;
  *)
    echo "Unsupported TARGET_LANG=${TARGET_LANG}" >&2
    exit 1
    ;;
esac
DIRECTION="${DIRECTION:-${DEFAULT_DIRECTION}}"
INPUT="${INPUT:-/projects/prudlab/formosan_parallel_corpora/splits_${FILE_SHORT}_v1/big_corpus_${FILE_SHORT}_${TIER}.csv}"
MODEL="${MODEL:?Set MODEL to a trained final/best model directory}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/formosan_mt_experiments/reports/${TIER}_${DIRECTION}_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUT_DIR}"
nvidia-smi || true

srun --cpu-bind=cores python -u "${EXP_DIR}/scripts/evaluate_directional.py" \
  --input "${INPUT}" \
  --tokenizer "${TOKENIZER}" \
  --model "${MODEL}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --direction "${DIRECTION}" \
  --output-csv "${OUT_DIR}/predictions.csv" \
  --output-json "${OUT_DIR}/metrics.json" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-length "${MAX_LENGTH:-384}" \
  --beam "${BEAM:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --device cuda
