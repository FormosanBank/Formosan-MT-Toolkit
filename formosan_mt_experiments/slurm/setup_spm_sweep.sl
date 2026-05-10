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
INPUT="${INPUT:-/projects/prudlab/formosan_parallel_corpora/big_corpus_en.csv}"
OUT_DIR="${OUT_DIR:-/scratch/scheppat/formosan_mt_experiments/data/tokenizer_sweep}"
SETUP_SCRIPT="${SETUP_SCRIPT:-/home/scheppat/nllb-scripts/setup_formosan_nllb200.py}"

mkdir -p "${OUT_DIR}"

python -u "${EXP_DIR}/scripts/setup_tokenizer_sweep.py" \
  --input "${INPUT}" \
  --setup-script "${SETUP_SCRIPT}" \
  --base-model "${BASE_MODEL:-facebook/nllb-200-distilled-600M}" \
  --output-dir "${OUT_DIR}" \
  --spm-vocabs "${SPM_VOCABS:-8192,16384,32768}" \
  --setup-splits "${SETUP_SPLITS:-train,validate,valid,val}" \
  --min-char-frequency 3 \
  --run-smoke \
  --samples-per-lang 1
