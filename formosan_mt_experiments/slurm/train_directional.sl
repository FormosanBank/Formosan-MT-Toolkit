#!/bin/bash
#SBATCH --job-name=formosan_mt_train
#SBATCH --partition=medium
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vr40g|vr80g|vr144g
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXP_DIR="${EXP_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRATCH="${SCRATCH:-${HOME}/formosan_mt_work}"
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
PROJECT_DATA="${PROJECT_DATA:-${EXP_DIR}/data/corpora}"
if [[ -n "${CORPUS_NAME:-}" ]]; then
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}/${CORPUS_NAME}}"
  DEFAULT_INPUT="${CORPUS_DIR}/big_corpus_${FILE_SHORT}_${TIER}.csv"
else
  CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DATA}}"
  DEFAULT_INPUT="${CORPUS_DIR}/splits_${FILE_SHORT}_v1/big_corpus_${FILE_SHORT}_${TIER}.csv"
fi
INPUT="${INPUT:-${DEFAULT_INPUT}}"
TOKENIZER="${TOKENIZER:?Set TOKENIZER to the prepared tokenizer directory}"
MODEL="${MODEL:?Set MODEL to the prepared base model directory}"
SETUP_MANIFEST="${SETUP_MANIFEST:?Set SETUP_MANIFEST to the setup manifest}"
CORPUS_MANIFEST="${CORPUS_MANIFEST:-${CORPUS_DIR}/provenance/mt_build_manifest.json}"
VALIDATION_REPORT="${VALIDATION_REPORT:-${CORPUS_DIR}/provenance/validate_${FILE_SHORT}_${TIER}_runtime.json}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/default_experiment.json}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/formosan_mt_experiments/runs/${TIER}_${DIRECTION}_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUT_DIR}"
nvidia-smi || true

train_command=(
  python -u "${EXP_DIR}/scripts/train_directional.py"
  --input "${INPUT}" \
  --tokenizer "${TOKENIZER}" \
  --model "${MODEL}" \
  --output-dir "${OUT_DIR}" \
  --profile "${PROFILE}" \
  --corpus-manifest "${CORPUS_MANIFEST}" \
  --validation-report "${VALIDATION_REPORT}" \
  --setup-manifest "${SETUP_MANIFEST}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --direction "${DIRECTION}" \
  --device cuda
)

add_override() {
  local variable="$1"
  local option="$2"
  if [[ -n "${!variable:-}" ]]; then
    train_command+=("${option}" "${!variable}")
  fi
}

add_override STEPS --steps
add_override BATCH_SIZE --batch-size
add_override GRAD_ACCUM_STEPS --grad-accum-steps
add_override MAX_LENGTH --max-length
add_override LEARNING_RATE --learning-rate
add_override WARMUP_STEPS --warmup-steps
add_override WEIGHT_DECAY --weight-decay
add_override MAX_GRAD_NORM --max-grad-norm
add_override LANGUAGE_SAMPLING_ALPHA --language-sampling-alpha
add_override SEED --seed
add_override SAVE_INTERVAL --save-interval
add_override EVAL_INTERVAL --eval-interval
add_override LOG_INTERVAL --log-interval
add_override EVAL_SAMPLES --eval-samples
add_override EVAL_BATCH_SIZE --eval-batch-size
add_override GENERATION_BATCH_SIZE --generation-batch-size
add_override VALIDATION_BEAM --validation-beam
add_override VALIDATION_METADATA_MODE --validation-metadata-mode
add_override VALIDATION_MAX_NEW_TOKENS --validation-max-new-tokens
add_override BEST_METRIC --best-metric
add_override EARLY_STOPPING_PATIENCE --early-stopping-patience
add_override EARLY_STOPPING_MIN_DELTA --early-stopping-min-delta
add_override EARLY_STOPPING_START_STEP --early-stopping-start-step
add_override RESUME_FROM --resume-from
add_override PRECISION --precision
add_override LOAD_DTYPE --load-dtype

if [[ -n "${GRADIENT_CHECKPOINTING:-}" ]]; then
  case "${GRADIENT_CHECKPOINTING}" in
    1|true|TRUE|yes|YES)
      train_command+=(--gradient-checkpointing)
      ;;
    0|false|FALSE|no|NO)
      train_command+=(--no-gradient-checkpointing)
      ;;
    *)
      echo "Invalid GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}" >&2
      exit 1
      ;;
  esac
fi

srun --cpu-bind=cores "${train_command[@]}"
