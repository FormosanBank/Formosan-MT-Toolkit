#!/bin/bash
#SBATCH --job-name=formosan_mt_eval
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
MODEL="${MODEL:?Set MODEL to a trained final/best model directory}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
CORPUS_MANIFEST="${CORPUS_MANIFEST:-${CORPUS_DIR}/provenance/mt_build_manifest.json}"
VALIDATION_REPORT="${VALIDATION_REPORT:-${CORPUS_DIR}/provenance/validate_${FILE_SHORT}_${TIER}_runtime.json}"
RUN_CONTRACT="${RUN_CONTRACT:?Set RUN_CONTRACT to the trainer run_contract.json}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/default_experiment.json}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/formosan_mt_experiments/reports/${TIER}_${DIRECTION}_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUT_DIR}"
nvidia-smi || true

evaluation_command=(
  python -u "${EXP_DIR}/scripts/evaluate_directional.py"
  --input "${INPUT}" \
  --tokenizer "${TOKENIZER}" \
  --model "${MODEL}" \
  --profile "${PROFILE}" \
  --corpus-manifest "${CORPUS_MANIFEST}" \
  --validation-report "${VALIDATION_REPORT}" \
  --run-contract "${RUN_CONTRACT}" \
  --target-lang "${TARGET_LANG}" \
  --target-col "${TARGET_COL}" \
  --direction "${DIRECTION}" \
  --output-csv "${OUT_DIR}/predictions.csv" \
  --output-json "${OUT_DIR}/metrics.json" \
  --device cuda
)

add_override() {
  local variable="$1"
  local option="$2"
  if [[ -n "${!variable:-}" ]]; then
    evaluation_command+=("${option}" "${!variable}")
  fi
}

add_override BATCH_SIZE --batch-size
add_override MAX_LENGTH --max-length
add_override BEAM --beam
add_override MAX_NEW_TOKENS --max-new-tokens
add_override MIN_NEW_TOKENS --min-new-tokens
add_override NO_REPEAT_NGRAM_SIZE --no-repeat-ngram-size
add_override REPETITION_PENALTY --repetition-penalty
add_override LENGTH_PENALTY --length-penalty
add_override METADATA_MODES --metadata-modes
add_override BOOTSTRAP_SAMPLES --bootstrap-samples
add_override BOOTSTRAP_SEED --bootstrap-seed
add_override BOOTSTRAP_WORKERS --bootstrap-workers
add_override LOAD_DTYPE --load-dtype

srun --cpu-bind=cores "${evaluation_command[@]}"
