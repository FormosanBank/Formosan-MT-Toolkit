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

EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat/projects/mt}"
export HF_HOME="${HF_HOME:-/scratch/scheppat/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
TARGET_LANG="${TARGET_LANG:-english}"
case "${TARGET_LANG}" in
  english)
    TARGET_COL="${TARGET_COL:-english_sentence}"
    FILE_SHORT="en"
    DEFAULT_OUT="${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep_spm8192"
    ;;
  chinese)
    TARGET_COL="${TARGET_COL:-chinese_sentence}"
    FILE_SHORT="zh"
    DEFAULT_OUT="${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep_zh_spm8192"
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
SETUP_SCRIPT="${SETUP_SCRIPT:-${EXP_DIR}/scripts/setup_formosan_nllb200.py}"
SETUP_SCRIPT_SHA256="${SETUP_SCRIPT_SHA256:-89bcb72d8c6b641ddce3a082f22750447cbe0530981ca177c9c91eac0084fa07}"

[[ -r "${SETUP_SCRIPT}" ]] || { echo "Missing NLLB setup implementation: ${SETUP_SCRIPT}" >&2; exit 1; }
actual_setup_sha256="$(sha256sum "${SETUP_SCRIPT}" | awk '{print $1}')"
if [[ "${actual_setup_sha256}" != "${SETUP_SCRIPT_SHA256}" ]]; then
  echo "NLLB setup implementation checksum mismatch: ${actual_setup_sha256}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

python -u "${EXP_DIR}/scripts/setup_tokenizer_sweep.py" \
  --input "${INPUT}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --setup-script "${SETUP_SCRIPT}" \
  --base-model "${BASE_MODEL:-facebook/nllb-200-distilled-600M}" \
  --output-dir "${OUT_DIR}" \
  --spm-vocabs "${SPM_VOCABS:-8192}" \
  --setup-splits "${SETUP_SPLITS:-train,validate,valid,val}" \
  --min-char-frequency 3 \
  --run-smoke \
  --samples-per-lang 1
