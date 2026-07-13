#!/bin/bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat/projects/mt}"
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
CORPUS_LABEL="${CORPUS_NAME:-legacy}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/formosan_mt_experiments/data/${CORPUS_LABEL}}"
RUNS_DIR="${RUNS_DIR:-${SCRATCH}/formosan_mt_experiments/runs/${CORPUS_LABEL}}"
REPORTS_DIR="${REPORTS_DIR:-${SCRATCH}/formosan_mt_experiments/reports/${CORPUS_LABEL}}"
JOBS_DIR="${JOBS_DIR:-/home/scheppat/jobs/mt}"
STATE_DIR="${STATE_DIR:-${JOBS_DIR}/submission_state_v1_spm8k_${CORPUS_LABEL}_${RUN_STAMP}}"

SETUP_SL="${SETUP_SL:-${EXP_DIR}/slurm/setup_spm_sweep.sl}"
TRAIN_SL="${TRAIN_SL:-${EXP_DIR}/slurm/train_directional.sl}"
EVAL_SL="${EVAL_SL:-${EXP_DIR}/slurm/evaluate_directional.sl}"

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

common_export() {
  local target_lang="$1"
  local direction="${2:-}"
  local pieces=(
    "EXP_DIR=${EXP_DIR}"
    "SCRATCH=${SCRATCH}"
    "PROJECT_DATA=${PROJECT_DATA}"
    "TARGET_LANG=${target_lang}"
  )
  if [[ -n "${CORPUS_NAME:-}" ]]; then
    pieces+=("CORPUS_NAME=${CORPUS_NAME}")
  fi
  if [[ -n "${direction}" ]]; then
    pieces+=("DIRECTION=${direction}")
  fi
  local joined
  joined="$(IFS=,; echo "${pieces[*]}")"
  printf 'ALL,%s' "${joined}"
}

submit_setup() {
  local target_lang="$1"
  local short="$2"
  local token_dir="${DATA_DIR}/tokenizer_sweep_${short}_spm8192"
  local tokenizer="${token_dir}/formosan_multilingual_nllb_spm8192_tokenizer"
  local model="${token_dir}/formosan_multilingual_nllb_spm8192_model"
  local label="setup_${short}_spm8192"
  if [[ -d "${tokenizer}" && -d "${model}" ]]; then
    echo "${label}=already_exists ${token_dir}"
    return 0
  fi
  submit_job "${label}" \
    --job-name="v1_${CORPUS_LABEL}_${short}_spm8192_setup" \
    --partition="${SETUP_PARTITION:-short}" \
    --time="${SETUP_TIME:-12:00:00}" \
    --cpus-per-task="${SETUP_CPUS:-16}" \
    --mem="${SETUP_MEM:-128G}" \
    --export="$(common_export "${target_lang}"),OUT_DIR=${token_dir},SPM_VOCABS=8192,SETUP_SPLITS=train,validate,BASE_MODEL=facebook/nllb-200-distilled-600M" \
    "${SETUP_SL}"
}

setup_dependency() {
  local short="$1"
  local label="setup_${short}_spm8192"
  local record="${STATE_DIR}/${label}.id"
  if [[ -s "${record}" ]]; then
    printf '%s' "--dependency=afterok:$(job_id "${label}")"
  fi
}

submit_direction() {
  local target_lang="$1"
  local short="$2"
  local direction="$3"
  local token_dir="${DATA_DIR}/tokenizer_sweep_${short}_spm8192"
  local tokenizer="${token_dir}/formosan_multilingual_nllb_spm8192_tokenizer"
  local model="${token_dir}/formosan_multilingual_nllb_spm8192_model"
  local run_out="${RUNS_DIR}/v1_spm8192_${direction}_${RUN_STAMP}"
  local train_label="train_${direction}"
  local setup_dep
  setup_dep="$(setup_dependency "${short}")"

  local -a train_args=(
    --job-name="v1_${CORPUS_LABEL}_${direction}"
    --partition="${TRAIN_PARTITION:-long}"
    --time="${TRAIN_TIME:-5-00:00:00}"
    --gres="${TRAIN_GRES:-gpu:1}"
    --constraint="${TRAIN_CONSTRAINT:-vr40g|vr80g|vr144g}"
    --cpus-per-task="${TRAIN_CPUS:-8}"
    --mem="${TRAIN_MEM:-128G}"
  )
  if [[ -n "${setup_dep}" ]]; then
    train_args+=("${setup_dep}")
  fi
  train_args+=(
    --export="$(common_export "${target_lang}" "${direction}"),TOKENIZER=${tokenizer},MODEL=${model},OUT_DIR=${run_out},STEPS=300000,BATCH_SIZE=16,GRAD_ACCUM_STEPS=4,MAX_LENGTH=384,LEARNING_RATE=2e-5,SAVE_INTERVAL=0,EVAL_INTERVAL=25000,EVAL_SAMPLES=256,EVAL_BATCH_SIZE=16"
    "${TRAIN_SL}"
  )
  submit_job "${train_label}" "${train_args[@]}"

  local train_id
  train_id="$(job_id "${train_label}")"
  for checkpoint in final best; do
    submit_job "eval_${direction}_${checkpoint}" \
      --job-name="v1_${CORPUS_LABEL}_${direction}_eval_${checkpoint}" \
      --partition="${EVAL_PARTITION:-medium}" \
      --time="${EVAL_TIME:-1-00:00:00}" \
      --gres="${EVAL_GRES:-gpu:1}" \
      --constraint="${EVAL_CONSTRAINT:-vr40g|vr80g|vr144g}" \
      --cpus-per-task="${EVAL_CPUS:-8}" \
      --mem="${EVAL_MEM:-96G}" \
      --dependency="afterok:${train_id}" \
      --export="$(common_export "${target_lang}" "${direction}"),MODEL=${run_out}/${checkpoint},TOKENIZER=${run_out}/${checkpoint},OUT_DIR=${REPORTS_DIR}/v1_spm8192_${direction}_${checkpoint}_${RUN_STAMP},BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256" \
      "${EVAL_SL}"
  done
}

echo "RUN_STAMP=${RUN_STAMP}"
echo "CORPUS_NAME=${CORPUS_NAME:-}"
echo "CORPUS_LABEL=${CORPUS_LABEL}"
echo "STATE_DIR=${STATE_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "REPORTS_DIR=${REPORTS_DIR}"

submit_setup english en
submit_setup chinese zh

submit_direction english en f2en
submit_direction english en en2f
submit_direction chinese zh f2zh
submit_direction chinese zh zh2f

touch "${STATE_DIR}/DONE"
echo "DONE_SUBMIT RUN_STAMP=${RUN_STAMP} CORPUS_LABEL=${CORPUS_LABEL}"
