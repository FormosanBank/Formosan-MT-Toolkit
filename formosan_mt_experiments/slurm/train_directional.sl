#!/bin/bash
#SBATCH --job-name=formosan_mt_train
#SBATCH --account=prudlab
#SBATCH --partition=medium
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vr40g|vr80g|vr144g
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

EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat/projects/mt}"
export HF_HOME="${HF_HOME:-/scratch/scheppat/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
TIER="${TIER:-in_domain_hard}"
TARGET_LANG="${TARGET_LANG:-english}"
case "${TARGET_LANG}" in
  english)
    TARGET_COL="${TARGET_COL:-english_sentence}"
    FILE_SHORT="en"
    DEFAULT_DIRECTION="f2en"
    DEFAULT_TOKEN_DIR="${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep_spm8192"
    DEFAULT_VOCAB="8192"
    ;;
  chinese)
    TARGET_COL="${TARGET_COL:-chinese_sentence}"
    FILE_SHORT="zh"
    DEFAULT_DIRECTION="f2zh"
    DEFAULT_TOKEN_DIR="${SCRATCH}/formosan_mt_experiments/data/tokenizer_sweep_zh_spm8192"
    DEFAULT_VOCAB="8192"
    ;;
  *)
    echo "Unsupported TARGET_LANG=${TARGET_LANG}" >&2
    exit 1
    ;;
esac
DIRECTION="${DIRECTION:-${DEFAULT_DIRECTION}}"
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
if [[ -n "${CORPUS_NAME:-}" ]]; then
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}/${CORPUS_NAME}}"
  DEFAULT_INPUT="${CORPUS_DIR}/big_corpus_${FILE_SHORT}_${TIER}.csv"
else
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}}"
  DEFAULT_INPUT="${CORPUS_DIR}/splits_${FILE_SHORT}_v1/big_corpus_${FILE_SHORT}_${TIER}.csv"
fi
INPUT="${INPUT:-${DEFAULT_INPUT}}"
TOKENIZER="${TOKENIZER:-${DEFAULT_TOKEN_DIR}/formosan_multilingual_nllb_spm${DEFAULT_VOCAB}_tokenizer}"
MODEL="${MODEL:-${DEFAULT_TOKEN_DIR}/formosan_multilingual_nllb_spm${DEFAULT_VOCAB}_model}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/formosan_mt_experiments/runs/E3_spm8192_${TIER}_${DIRECTION}_$(date +%Y%m%d-%H%M%S)}"

if [[ "${USE_DAE:-0}" == "1" ]]; then
  MODEL="${DAE_MODEL:-${SCRATCH}/formosan_mt_experiments/runs/latest_dae/final}"
fi

LORA_ARGS=()
if [[ "${LORA_R:-0}" != "0" ]]; then
  LORA_ARGS+=(--lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA:-32}" --lora-dropout "${LORA_DROPOUT:-0.05}")
fi

mkdir -p "${OUT_DIR}"
nvidia-smi || true

srun --cpu-bind=cores python -u "${EXP_DIR}/scripts/train_directional_nllb.py" \
  --input "${INPUT}" \
  --tokenizer "${TOKENIZER}" \
  --model "${MODEL}" \
  --output-dir "${OUT_DIR}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --direction "${DIRECTION}" \
  --steps "${STEPS:-300000}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}" \
  --max-length "${MAX_LENGTH:-384}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --save-interval "${SAVE_INTERVAL:-25000}" \
  --eval-interval "${EVAL_INTERVAL:-25000}" \
  --log-interval "${LOG_INTERVAL:-1000}" \
  --eval-samples "${EVAL_SAMPLES:-256}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-16}" \
  --validation-beam "${VALIDATION_BEAM:-2}" \
  --validation-max-new-tokens "${VALIDATION_MAX_NEW_TOKENS:-256}" \
  --best-metric "${BEST_METRIC:-chrF2}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA:-0.05}" \
  --early-stopping-start-step "${EARLY_STOPPING_START_STEP:-30000}" \
  --resume-from "${RESUME_FROM:-auto}" \
  --precision bf16 \
  --device cuda \
  "${LORA_ARGS[@]}"
