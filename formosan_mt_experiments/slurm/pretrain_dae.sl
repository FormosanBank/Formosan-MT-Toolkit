#!/bin/bash
#SBATCH --job-name=formosan_mt_dae
#SBATCH --account=prudlab
#SBATCH --partition=medium
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXP_DIR="${EXP_DIR:-/home/scheppat/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat}"
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
TIER="${TIER:-in_domain_hard}"
INPUT="${INPUT:-${SCRATCH}/formosan_mt_experiments/data/splits_en_v1/big_corpus_en_${TIER}.csv}"
TOKENIZER="${TOKENIZER:-${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_tokenizer}"
MODEL="${MODEL:-${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_model}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/formosan_mt_experiments/runs/E2_dae_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUT_DIR}"
nvidia-smi || true

srun --cpu-bind=cores python -u "${EXP_DIR}/scripts/pretrain_dae_nllb.py" \
  --input "${INPUT}" \
  --tokenizer "${TOKENIZER}" \
  --model "${MODEL}" \
  --output-dir "${OUT_DIR}" \
  --steps "${STEPS:-100000}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}" \
  --max-length "${MAX_LENGTH:-384}" \
  --learning-rate "${LEARNING_RATE:-3e-5}" \
  --save-interval "${SAVE_INTERVAL:-25000}" \
  --eval-interval "${EVAL_INTERVAL:-25000}" \
  --log-interval "${LOG_INTERVAL:-1000}" \
  --eval-samples "${EVAL_SAMPLES:-256}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --precision bf16 \
  --device cuda
