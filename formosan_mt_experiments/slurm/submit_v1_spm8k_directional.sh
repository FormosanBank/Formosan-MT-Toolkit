#!/bin/bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat/projects/mt}"
PROJECT_DATA="${PROJECT_DATA:-/projects/prudlab/formosan_parallel_corpora}"
: "${CORPUS_NAME:?Set CORPUS_NAME to a named corpus such as public_no_bible or private_no_bible}"
CORPUS_LABEL="${CORPUS_NAME}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/formosan_mt_experiments/data/${CORPUS_LABEL}}"
RUNS_DIR="${RUNS_DIR:-${SCRATCH}/formosan_mt_experiments/runs/${CORPUS_LABEL}}"
REPORTS_DIR="${REPORTS_DIR:-${SCRATCH}/formosan_mt_experiments/reports/${CORPUS_LABEL}}"
LOGS_DIR="${LOGS_DIR:-${SCRATCH}/formosan_mt_experiments/logs/${RUN_STAMP}}"
JOBS_DIR="${JOBS_DIR:-/home/scheppat/jobs/mt}"
STATE_DIR="${STATE_DIR:-${JOBS_DIR}/submission_state_v1_spm8k_${CORPUS_LABEL}_${RUN_STAMP}}"
MANIFEST_DIR="${MANIFEST_DIR:-/projects/prudlab/formosan_mt_experiments}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/default_experiment.json}"
SETUP_IMPLEMENTATION="${SETUP_IMPLEMENTATION:-${EXP_DIR}/scripts/setup_formosan_nllb200.py}"
SETUP_IMPLEMENTATION_SHA256="${SETUP_IMPLEMENTATION_SHA256:-c0626e187bf33043effb2bdf6f0af9f55b99eb0a0e8763f1805fedb0a7a7f4a3}"

SETUP_SL="${SETUP_SL:-${EXP_DIR}/slurm/setup_spm_sweep.sl}"
TRAIN_SL="${TRAIN_SL:-${EXP_DIR}/slurm/train_directional.sl}"
EVAL_SL="${EVAL_SL:-${EXP_DIR}/slurm/evaluate_directional.sl}"
VALIDATE_SL="${VALIDATE_SL:-${EXP_DIR}/slurm/validate_corpus.sl}"

mkdir -p "${STATE_DIR}" "${LOGS_DIR}"

for short in en zh; do
  input="${PROJECT_DATA}/${CORPUS_NAME}/big_corpus_${short}_in_domain_hard.csv"
  provenance="${PROJECT_DATA}/${CORPUS_NAME}/provenance/mt_build_manifest.json"
  [[ -r "${input}" ]] || { echo "Missing corpus input: ${input}" >&2; exit 1; }
  [[ -r "${provenance}" ]] || { echo "Missing corpus provenance: ${provenance}" >&2; exit 1; }
done
[[ -r "${PROFILE}" ]] || { echo "Missing experiment profile: ${PROFILE}" >&2; exit 1; }
[[ -r "${SETUP_IMPLEMENTATION}" ]] || {
  echo "Missing versioned NLLB setup implementation: ${SETUP_IMPLEMENTATION}" >&2
  exit 1
}
actual_setup_sha256="$(sha256sum "${SETUP_IMPLEMENTATION}" | awk '{print $1}')"
if [[ "${actual_setup_sha256}" != "${SETUP_IMPLEMENTATION_SHA256}" ]]; then
  echo "NLLB setup implementation checksum mismatch: ${actual_setup_sha256}" >&2
  exit 1
fi

submit_job() {
  local label="$1"
  shift
  local record="${STATE_DIR}/${label}.id"
  local out
  if [[ -s "${record}" ]]; then
    out="$(<"${record}")"
    local state
    state="$(sacct -j "${out}" --starttime 2020-01-01 --noheader --parsable2 --allocations --format=State 2>/dev/null | awk -F'|' 'NF && $1 != "" {print $1; exit}')"
    case "${state}" in
      PENDING*|RUNNING*|CONFIGURING*|COMPLETING*|COMPLETED*)
        echo "${label}=${out} (existing state=${state})"
        return 0
        ;;
      FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|PREEMPTED*|BOOT_FAIL*|DEADLINE*)
        echo "${label}=${out} terminal state=${state}; resubmitting" >&2
        rm -f "${record}"
        ;;
      *)
        echo "Cannot safely classify recorded job ${out} for ${label}; sacct state=${state:-unknown}." >&2
        exit 1
        ;;
    esac
  fi
  out="$(sbatch --parsable \
    --output="${LOGS_DIR}/%x-%j.out" \
    --error="${LOGS_DIR}/%x-%j.err" \
    "$@")"
  printf '%s\n' "${out}" > "${record}"
  echo "${label}=${out}"
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
    "PROFILE=${PROFILE}"
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
  local setup_manifest="${token_dir}/formosan_multilingual_nllb_spm8192_setup_manifest.json"
  local label="setup_${short}_spm8192"
  if [[ -f "${tokenizer}/tokenizer_config.json" && -f "${tokenizer}/sentencepiece.bpe.model" \
      && -f "${model}/config.json" && -f "${setup_manifest}" ]] \
      && compgen -G "${model}/model*.safetensors" > /dev/null; then
    if python - "${setup_manifest}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("complete") is True else 1)
PY
    then
      echo "${label}=already_exists ${token_dir}"
      return 0
    fi
  fi
  submit_job "${label}" \
    --job-name="v1_${CORPUS_LABEL}_${short}_spm8192_setup" \
    --partition="${SETUP_PARTITION:-short}" \
    --time="${SETUP_TIME:-12:00:00}" \
    --cpus-per-task="${SETUP_CPUS:-16}" \
    --mem="${SETUP_MEM:-128G}" \
    --dependency="afterok:$(job_id "validate_${short}")" \
    --export="$(common_export "${target_lang}"),OUT_DIR=${token_dir},SPM_VOCABS=8192,SETUP_SCRIPT=${SETUP_IMPLEMENTATION},SETUP_SCRIPT_SHA256=${SETUP_IMPLEMENTATION_SHA256}" \
    "${SETUP_SL}"
}

submit_validation() {
  local target_lang="$1"
  local short="$2"
  submit_job "validate_${short}" \
    --job-name="v1_${CORPUS_LABEL}_${short}_corpus_validate" \
    --partition="${VALIDATE_PARTITION:-short}" \
    --time="${VALIDATE_TIME:-02:00:00}" \
    --cpus-per-task="${VALIDATE_CPUS:-8}" \
    --mem="${VALIDATE_MEM:-64G}" \
    --export="$(common_export "${target_lang}"),MIN_TEST_RATIO=0.075,MIN_VALIDATE_RATIO=0.025" \
    "${VALIDATE_SL}"
}

setup_dependency() {
  local short="$1"
  local label="setup_${short}_spm8192"
  local record="${STATE_DIR}/${label}.id"
  if [[ -s "${record}" ]]; then
    printf '%s' "--dependency=afterok:$(job_id "${label}")"
  else
    printf '%s' "--dependency=afterok:$(job_id "validate_${short}")"
  fi
}

submit_direction() {
  local target_lang="$1"
  local short="$2"
  local direction="$3"
  local token_dir="${DATA_DIR}/tokenizer_sweep_${short}_spm8192"
  local tokenizer="${token_dir}/formosan_multilingual_nllb_spm8192_tokenizer"
  local model="${token_dir}/formosan_multilingual_nllb_spm8192_model"
  local setup_manifest="${token_dir}/formosan_multilingual_nllb_spm8192_setup_manifest.json"
  local corpus_manifest="${PROJECT_DATA}/${CORPUS_NAME}/provenance/mt_build_manifest.json"
  local validation_report="${PROJECT_DATA}/${CORPUS_NAME}/provenance/validate_${short}_in_domain_hard_runtime.json"
  local run_out="${RUNS_DIR}/v1_spm8192_${direction}_${RUN_STAMP}"
  local train_label="train_${direction}"
  local setup_dep
  setup_dep="$(setup_dependency "${short}")"

  local -a train_args=(
    --job-name="v1_${CORPUS_LABEL}_${direction}"
    --partition="${TRAIN_PARTITION:-medium}"
    --time="${TRAIN_TIME:-2-00:00:00}"
    --gres="${TRAIN_GRES:-gpu:1}"
    --constraint="${TRAIN_CONSTRAINT:-vr40g|vr80g|vr144g}"
    --cpus-per-task="${TRAIN_CPUS:-8}"
    --mem="${TRAIN_MEM:-128G}"
  )
  if [[ -n "${setup_dep}" ]]; then
    train_args+=("${setup_dep}")
  fi
  train_args+=(
    --export="$(common_export "${target_lang}" "${direction}"),TOKENIZER=${tokenizer},MODEL=${model},SETUP_MANIFEST=${setup_manifest},CORPUS_MANIFEST=${corpus_manifest},VALIDATION_REPORT=${validation_report},OUT_DIR=${run_out}"
    "${TRAIN_SL}"
  )
  submit_job "${train_label}" "${train_args[@]}"

  local train_id
  train_id="$(job_id "${train_label}")"
  local train_state
  train_state="$(sacct -j "${train_id}" --starttime 2020-01-01 --noheader --parsable2 --allocations --format=State 2>/dev/null | awk -F'|' 'NF && $1 != "" {print $1; exit}')"
  local -a eval_dependency=()
  case "${train_state}" in
    COMPLETED*)
      # Slurm can reject afterok dependencies once a completed parent leaves
      # controller memory, even though sacct still retains its terminal state.
      ;;
    PENDING*|RUNNING*|CONFIGURING*|COMPLETING*)
      eval_dependency+=(--dependency="afterok:${train_id}")
      ;;
    *)
      echo "Cannot submit evaluations for trainer ${train_id}; state=${train_state:-unknown}." >&2
      exit 1
      ;;
  esac
  for checkpoint in final best; do
    submit_job "eval_${direction}_${checkpoint}" \
      --job-name="v1_${CORPUS_LABEL}_${direction}_eval_${checkpoint}" \
      --partition="${EVAL_PARTITION:-medium}" \
      --time="${EVAL_TIME:-1-00:00:00}" \
      --gres="${EVAL_GRES:-gpu:1}" \
      --constraint="${EVAL_CONSTRAINT:-vr40g|vr80g|vr144g}" \
      --cpus-per-task="${EVAL_CPUS:-8}" \
      --mem="${EVAL_MEM:-96G}" \
      "${eval_dependency[@]}" \
      --export="$(common_export "${target_lang}" "${direction}"),MODEL=${run_out}/${checkpoint},TOKENIZER=${run_out}/${checkpoint},CORPUS_MANIFEST=${corpus_manifest},VALIDATION_REPORT=${validation_report},RUN_CONTRACT=${run_out}/run_contract.json,OUT_DIR=${REPORTS_DIR}/v1_spm8192_${direction}_${checkpoint}_${RUN_STAMP}" \
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
echo "LOGS_DIR=${LOGS_DIR}"

submit_validation english en
submit_validation chinese zh

submit_setup english en
submit_setup chinese zh

submit_direction english en f2en
submit_direction english en en2f
submit_direction chinese zh f2zh
submit_direction chinese zh zh2f

GIT_COMMIT="${GIT_COMMIT:-$(git -C "${EXP_DIR}" rev-parse HEAD 2>/dev/null || printf unknown)}"
python -u "${EXP_DIR}/scripts/write_submission_manifest.py" \
  --corpus-name "${CORPUS_NAME}" \
  --run-stamp "${RUN_STAMP}" \
  --git-commit "${GIT_COMMIT}" \
  --state-dir "${STATE_DIR}" \
  --project-data "${PROJECT_DATA}" \
  --profile "${PROFILE}" \
  --experiment-root "${EXP_DIR}" \
  --output "${MANIFEST_DIR}/submission_manifest_v1_spm8k_${CORPUS_NAME}_${RUN_STAMP}.json"
touch "${STATE_DIR}/DONE"
echo "DONE_SUBMIT RUN_STAMP=${RUN_STAMP} CORPUS_LABEL=${CORPUS_LABEL}"
