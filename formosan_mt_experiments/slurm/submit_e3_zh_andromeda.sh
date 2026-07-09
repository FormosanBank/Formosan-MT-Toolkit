#!/bin/bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat}"
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/formosan_mt_experiments/data}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DATA}/splits_zh_v1}"
INPUT="${INPUT:-${SPLIT_DIR}/big_corpus_zh_in_domain_hard.csv}"
TOKEN_DIR="${TOKEN_DIR:-${DATA_DIR}/tokenizer_sweep_zh_spm8192}"
TOKENIZER="${TOKENIZER:-${TOKEN_DIR}/formosan_multilingual_nllb_spm8192_tokenizer}"
MODEL="${MODEL:-${TOKEN_DIR}/formosan_multilingual_nllb_spm8192_model}"
JOBS_DIR="${JOBS_DIR:-/home/scheppat/jobs/mt}"
STATE_DIR="${STATE_DIR:-${JOBS_DIR}/submission_state_zh_e3_${RUN_STAMP}}"

BUILD_SL="${BUILD_SL:-${JOBS_DIR}/formosan_exp_build_splits.sl}"
SETUP_SL="${SETUP_SL:-${JOBS_DIR}/formosan_exp_setup_spm_sweep.sl}"
TRAIN_SL="${TRAIN_SL:-${JOBS_DIR}/formosan_exp_train_directional.sl}"
EVAL_SL="${EVAL_SL:-${JOBS_DIR}/formosan_exp_evaluate_directional.sl}"

mkdir -p "${STATE_DIR}"

submit_job() {
  local label="$1"
  shift
  local record="${STATE_DIR}/${label}.id"
  local out
  if [[ -s "${record}" ]]; then
    out="$(<"${record}")"
    echo "${label}=${out} (existing)"
  else
    out="$(sbatch --parsable "$@")"
    printf '%s\n' "${out}" > "${record}"
    echo "${label}=${out}"
  fi
}

job_id() {
  local label="$1"
  cat "${STATE_DIR}/${label}.id"
}

echo "RUN_STAMP=${RUN_STAMP}"
echo "STATE_DIR=${STATE_DIR}"
echo "INPUT=${INPUT}"
echo "TOKENIZER=${TOKENIZER}"

BUILD_DEP=()
if [[ ! -s "${INPUT}" ]]; then
  submit_job build_zh_splits \
    --job-name=E3_zh_build_splits \
    --partition=short --time=02:00:00 --cpus-per-task=8 --mem=32G \
    --export=ALL,EXP_DIR="${EXP_DIR}",TARGET_LANG=chinese,INPUT="${PROJECT_DATA}/big_corpus_zh.csv",OUT_DIR="${SPLIT_DIR}",OUTPUT_PREFIX=big_corpus_zh,MIN_FORMOSAN_TOKENS=4,MIN_TARGET_TOKENS=4 \
    "${BUILD_SL}"
  BUILD_DEP=(--dependency="afterok:$(job_id build_zh_splits)")
else
  echo "split_exists=${INPUT}"
fi

SETUP_DEP=("${BUILD_DEP[@]}")
if [[ ! -d "${TOKENIZER}" || ! -d "${MODEL}" ]]; then
  submit_job setup_zh_spm8192 \
    --job-name=E3_zh_setup_spm8192 \
    --partition=short --time=12:00:00 --cpus-per-task=16 --mem=128G \
    "${SETUP_DEP[@]}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",TARGET_LANG=chinese,INPUT="${INPUT}",OUT_DIR="${TOKEN_DIR}",SETUP_SCRIPT=/home/scheppat/nllb-scripts/setup_formosan_nllb200.py,SPM_VOCABS=8192,SETUP_SPLITS=train,validate,BASE_MODEL=facebook/nllb-200-distilled-600M \
    "${SETUP_SL}"
  TRAIN_DEP=(--dependency="afterok:$(job_id setup_zh_spm8192)")
else
  echo "tokenizer_model_exist=${TOKEN_DIR}"
  TRAIN_DEP=("${BUILD_DEP[@]}")
fi

for DIRN in f2zh zh2f; do
  RUN_OUT="${SCRATCH}/formosan_mt_experiments/runs/E3_zh_spm8192_${DIRN}_${RUN_STAMP}"
  TRAIN_LABEL="E3_zh_${DIRN}_train"
  submit_job "${TRAIN_LABEL}" \
    --job-name="E3_zh_${DIRN}" \
    --partition=medium --time=2-00:00:00 --gres=gpu:1 --constraint=vr80g\|vr144g --cpus-per-task=8 --mem=128G \
    "${TRAIN_DEP[@]}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TARGET_LANG=chinese,TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",TOKENIZER="${TOKENIZER}",MODEL="${MODEL}",OUT_DIR="${RUN_OUT}",STEPS=300000,BATCH_SIZE=16,GRAD_ACCUM_STEPS=4,MAX_LENGTH=384,LEARNING_RATE=2e-5,SAVE_INTERVAL=25000,EVAL_INTERVAL=25000,EVAL_SAMPLES=256,EVAL_BATCH_SIZE=16 \
    "${TRAIN_SL}"
  TRAIN_ID="$(job_id "${TRAIN_LABEL}")"

  submit_job "E3_zh_${DIRN}_eval_final" \
    --job-name="E3_zh_${DIRN}_eval_final" \
    --partition=medium --time=08:00:00 --gres=gpu:1 --constraint=vr80g\|vr144g --cpus-per-task=8 --mem=96G \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TARGET_LANG=chinese,TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/final",TOKENIZER="${RUN_OUT}/final",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E3_zh_spm8192_${DIRN}_final_${RUN_STAMP}",BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
    "${EVAL_SL}"

  submit_job "E3_zh_${DIRN}_eval_best" \
    --job-name="E3_zh_${DIRN}_eval_best" \
    --partition=medium --time=08:00:00 --gres=gpu:1 --constraint=vr80g\|vr144g --cpus-per-task=8 --mem=96G \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TARGET_LANG=chinese,TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/best",TOKENIZER="${RUN_OUT}/best",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E3_zh_spm8192_${DIRN}_best_${RUN_STAMP}",BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
    "${EVAL_SL}"
done

touch "${STATE_DIR}/DONE"
echo "DONE_SUBMIT RUN_STAMP=${RUN_STAMP}"
