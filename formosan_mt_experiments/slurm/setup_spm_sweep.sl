#!/bin/bash
#SBATCH --job-name=formosan_mt_spm
#SBATCH --account=prudlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/scheppat/logs/%x-%j.out
#SBATCH --error=/home/scheppat/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

EXP_DIR="${EXP_DIR:-/home/scheppat/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat}"
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
TARGET_LANG="${TARGET_LANG:-english}"
case "${TARGET_LANG}" in
  english)
    TARGET_COL="${TARGET_COL:-english_sentence}"
    FILE_SHORT="en"
    DEFAULT_OUT="${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep"
    ;;
  chinese)
    TARGET_COL="${TARGET_COL:-chinese_sentence}"
    FILE_SHORT="zh"
    DEFAULT_OUT="${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep_zh"
    ;;
  *)
    echo "Unsupported TARGET_LANG=${TARGET_LANG}" >&2
    exit 1
    ;;
esac
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
if [[ -n "${CORPUS_NAME:-}" ]]; then
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}/${CORPUS_NAME}}"
  DEFAULT_INPUT="${CORPUS_DIR}/big_corpus_${FILE_SHORT}_in_domain_hard.csv"
else
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}}"
  DEFAULT_INPUT="${CORPUS_DIR}/splits_${FILE_SHORT}_v1/big_corpus_${FILE_SHORT}_in_domain_hard.csv"
fi
INPUT="${INPUT:-${DEFAULT_INPUT}}"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT}}"
SETUP_SCRIPT="${SETUP_SCRIPT:-/home/scheppat/nllb-scripts/setup_formosan_nllb200.py}"

mkdir -p "${OUT_DIR}"

python -u "${EXP_DIR}/scripts/setup_tokenizer_sweep.py" \
  --input "${INPUT}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --setup-script "${SETUP_SCRIPT}" \
  --base-model "${BASE_MODEL:-facebook/nllb-200-distilled-600M}" \
  --output-dir "${OUT_DIR}" \
  --spm-vocabs "${SPM_VOCABS:-8192,16384,32768}" \
  --setup-splits "${SETUP_SPLITS:-train,validate,valid,val}" \
  --min-char-frequency 3 \
  --run-smoke \
  --samples-per-lang 1
