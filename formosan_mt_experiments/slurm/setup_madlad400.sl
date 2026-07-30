#!/bin/bash
#SBATCH --job-name=formosan_mt_madlad_setup
#SBATCH --account=prudlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/home/scheppat/logs/%x-%j.out
#SBATCH --error=/home/scheppat/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
CORPUS_NAME="${CORPUS_NAME:?CORPUS_NAME is required}"
CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}/${CORPUS_NAME}}"
INPUT_EN="${INPUT_EN:-${CORPUS_DIR}/big_corpus_en_in_domain_hard.csv}"
INPUT_ZH="${INPUT_ZH:-${CORPUS_DIR}/big_corpus_zh_in_domain_hard.csv}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/madlad400_3b_native.json}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

export HF_HOME="${HF_HOME:-/scratch/scheppat/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${OUT_DIR}"

srun --cpu-bind=cores python -u \
  "${EXP_DIR}/scripts/setup_formosan_madlad400.py" \
  --input-en "${INPUT_EN}" \
  --input-zh "${INPUT_ZH}" \
  --profile "${PROFILE}" \
  --output-dir "${OUT_DIR}"
