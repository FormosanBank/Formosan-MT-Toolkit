#!/bin/bash
#SBATCH --job-name=E0_legacy_multilingual
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

HOME_DIR="${HOME_DIR:-/home/scheppat}"
SCRATCH_DIR="${SCRATCH:-/scratch/scheppat}"
EXP_DIR="${EXP_DIR:-${HOME_DIR}/formosan_mt_experiments}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${SCRATCH_DIR}/temp"

TGT_LANG="${TGT_LANG:-english}"
case "${TGT_LANG}" in
  english) TGT_LID="eng_Latn" ;;
  chinese) TGT_LID="zho_Hant" ;;
  *) echo "ERROR: Unsupported TGT_LANG='${TGT_LANG}'"; exit 1 ;;
esac

INPUT="${INPUT:-${SCRATCH_DIR}/formosan_mt_experiments/data/splits_en_v1/big_corpus_en_in_domain_hard.csv}"
TOKENIZER_DIR="${TOKENIZER:-${SCRATCH_DIR}/nllb200-multilingual/tokenizer/formosan_multilingual_nllb_tokenizer}"
BASE_MODEL_DIR="${MODEL:-${SCRATCH_DIR}/nllb200-multilingual/model/formosan_multilingual_nllb_model}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-${HOME_DIR}/nllb-scripts/train_formosan_multilingual_nllb200.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${HOME_DIR}/nllb-scripts/eval_formosan_multilingual_nllb200.py}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-${SCRATCH_DIR}/formosan_mt_experiments/runs/E0_legacy_multilingual_bidir_${RUN_STAMP}}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Legacy baseline defaults: old multilingual setup, old tokenizer/model, same new
# hard split. Keep the late-April legacy sampling knobs that were already in use
# before E1-E4: alpha=0.5 and easy-source downweighting. Save/eval intervals are
# aligned so best validation checkpoints can be evaluated post-training without
# keeping 60+ checkpoints.
TEMPERATURE="${TEMPERATURE:-5.0}"
STEPS="${STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
CLIP_THRESHOLD="${CLIP_THRESHOLD:-1.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25000}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-256}"
EVAL_BEAMS="${EVAL_BEAMS:-4}"
P_SRC2TGT="${P_SRC2TGT:-0.5}"
EASY_SOURCE_WEIGHT="${EASY_SOURCE_WEIGHT:-0.1}"
ALPHA="${ALPHA:-0.5}"
NORMALIZE="${NORMALIZE:-1}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_BEAM="${EVAL_BEAM:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

echo "============================================================"
echo "E0 Legacy Multilingual Baseline"
echo "Job ID        : ${SLURM_JOB_ID:-manual}"
echo "Input         : ${INPUT}"
echo "Tokenizer     : ${TOKENIZER_DIR}"
echo "Base model    : ${BASE_MODEL_DIR}"
echo "Output        : ${OUT_DIR}"
echo "Target        : ${TGT_LANG} (${TGT_LID})"
echo "p(src->target): ${P_SRC2TGT}"
echo "Date          : $(date)"
echo "============================================================"
nvidia-smi || true

for path in "${TRAINING_SCRIPT}" "${EVAL_SCRIPT}" "${TOKENIZER_DIR}" "${BASE_MODEL_DIR}" "${INPUT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: Missing path: ${path}"
    exit 1
  fi
done

LOCAL_CORPUS="${SCRATCH_DIR}/temp/E0_${SLURM_JOB_ID:-$$}_$(basename "${INPUT}")"
cp -f "${INPUT}" "${LOCAL_CORPUS}"
trap 'rm -f "${LOCAL_CORPUS}"' EXIT

EXTRA_NORMALIZE=()
if [[ "${NORMALIZE}" -eq 1 ]]; then
  EXTRA_NORMALIZE+=(--normalize)
fi

EXTRA_SAMPLING=()
if [[ -n "${ALPHA}" ]]; then
  EXTRA_SAMPLING+=(--alpha "${ALPHA}")
fi
EXTRA_SAMPLING+=(--easy-source-weight "${EASY_SOURCE_WEIGHT}")

TRAIN_ARGS=(
  --multilingual
  --tgt-lang "${TGT_LANG}"
  --temperature "${TEMPERATURE}"
  --tokenizer "${TOKENIZER_DIR}"
  --model "${BASE_MODEL_DIR}"
  --input "${LOCAL_CORPUS}"
  --output-dir "${OUT_DIR}"
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-steps "${WARMUP_STEPS}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-threshold "${CLIP_THRESHOLD}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-samples "${EVAL_SAMPLE_SIZE}"
  --eval-beams "${EVAL_BEAMS}"
  --p-src2tgt "${P_SRC2TGT}"
  --fp16
  --device cuda
  "${EXTRA_SAMPLING[@]}"
  "${EXTRA_NORMALIZE[@]}"
)

echo "Training command:"
printf ' %q' "${CONDA_PREFIX}/bin/python" "${TRAINING_SCRIPT}" "${TRAIN_ARGS[@]}"
echo

set +e
srun --cpu-bind=cores "${CONDA_PREFIX}/bin/python" "${TRAINING_SCRIPT}" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOGS_DIR}/training.log"
train_status=${PIPESTATUS[0]}
set -e

if [[ ${train_status} -ne 0 ]]; then
  echo "Training failed (exit ${train_status}). Check ${LOGS_DIR}/training.log"
  exit "${train_status}"
fi

run_eval() {
  local label="$1"
  local model_dir="$2"
  local report_dir="${SCRATCH_DIR}/formosan_mt_experiments/reports/E0_legacy_multilingual_${label}_${RUN_STAMP}"
  mkdir -p "${report_dir}"
  if [[ ! -d "${model_dir}" ]]; then
    echo "Skipping ${label} eval; missing model dir: ${model_dir}"
    return 0
  fi

  local eval_args=(
    --multilingual
    --tgt-lang "${TGT_LANG}"
    --tokenizer "${model_dir}"
    --model "${model_dir}"
    --input "${LOCAL_CORPUS}"
    --tgt-lid "${TGT_LID}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --max-length "${MAX_LENGTH}"
    --beam "${EVAL_BEAM}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --csv-out "${report_dir}/predictions.csv"
    --save-json "${report_dir}/metrics.json"
  )
  if [[ "${NORMALIZE}" -eq 1 ]]; then
    eval_args+=(--normalize)
  else
    eval_args+=(--no-normalize)
  fi

  echo "------------------------------------------------------------"
  echo "Evaluating ${label}: ${model_dir}"
  echo "Report dir: ${report_dir}"
  echo "------------------------------------------------------------"
  set +e
  srun --cpu-bind=cores "${CONDA_PREFIX}/bin/python" "${EVAL_SCRIPT}" "${eval_args[@]}" \
    2>&1 | tee "${LOGS_DIR}/evaluation_${label}.log"
  local eval_status=${PIPESTATUS[0]}
  set -e
  if [[ ${eval_status} -ne 0 ]]; then
    echo "Evaluation ${label} failed (exit ${eval_status})."
  fi
}

run_eval "final" "${OUT_DIR}/final"

BEST_STEP="$(
  "${CONDA_PREFIX}/bin/python" - "${LOGS_DIR}/training.log" "${EVAL_INTERVAL}" <<'PY'
import re
import sys

log_path = sys.argv[1]
eval_interval = int(sys.argv[2])
pat = re.compile(r"\[Eval Global\]\s+mean token-loss\s+src->[^:]+:\s+([0-9.]+)\s+\|\s+[^:]+->Formosan:\s+([0-9.]+)")
fallback = re.compile(r"\[Eval Global\]\s+mean token-loss\s+src->[^:]+:\s+([0-9.]+)\s+\|\s+[^:]+->src:\s+([0-9.]+)")
rows = []
with open(log_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line) or fallback.search(line)
        if not m:
            continue
        idx = len(rows) + 1
        step = idx * eval_interval
        f_loss = float(m.group(1))
        b_loss = float(m.group(2))
        rows.append((step, (f_loss + b_loss) / 2.0, f_loss, b_loss))
if rows:
    best = min(rows, key=lambda x: x[1])
    print(best[0])
PY
)"

if [[ -n "${BEST_STEP}" ]]; then
  BEST_CKPT="${OUT_DIR}/checkpoints/step-$(printf '%06d' "${BEST_STEP}")"
  run_eval "best_avg_step_${BEST_STEP}" "${BEST_CKPT}"
else
  echo "No parseable validation loss found; skipping best checkpoint eval."
fi

echo "============================================================"
echo "E0 finished at: $(date)"
echo "Artifacts: ${OUT_DIR}"
echo "============================================================"
