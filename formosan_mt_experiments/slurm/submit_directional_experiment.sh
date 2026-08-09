#!/bin/bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH="${SCRATCH:-${HOME}/formosan_mt_work}"
PROJECT_DATA="${PROJECT_DATA:-${EXP_DIR}/data/corpora}"
: "${CORPUS_NAME:?Set CORPUS_NAME to public_no_bible or private_no_bible}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/default_experiment.json}"

readarray -t profile_values < <(
  python - "${PROFILE}" <<'PY'
import json
import re
import sys

profile = json.load(open(sys.argv[1], encoding="utf-8"))
family = str(profile.get("model_family") or "nllb").strip().lower()
recipe = str(profile["recipe_id"])
slug = re.sub(r"[^a-z0-9]+", "_", recipe.lower()).strip("_")
print(family)
print(recipe)
print(slug)
PY
)
MODEL_FAMILY="${profile_values[0]}"
RECIPE_ID="${profile_values[1]}"
RECIPE_SLUG="${profile_values[2]}"
case "${MODEL_FAMILY}" in
  nllb|madlad400) ;;
  *) echo "Unsupported model family: ${MODEL_FAMILY}" >&2; exit 1 ;;
esac

DATA_DIR="${DATA_DIR:-${SCRATCH}/formosan_mt_experiments/data/${CORPUS_NAME}}"
RUNS_DIR="${RUNS_DIR:-${SCRATCH}/formosan_mt_experiments/runs/${CORPUS_NAME}}"
REPORTS_DIR="${REPORTS_DIR:-${SCRATCH}/formosan_mt_experiments/reports/${CORPUS_NAME}}"
LOGS_DIR="${LOGS_DIR:-${SCRATCH}/formosan_mt_experiments/logs/${RUN_STAMP}}"
JOBS_DIR="${JOBS_DIR:-${SCRATCH}/formosan_mt_experiments/jobs}"
STATE_DIR="${STATE_DIR:-${JOBS_DIR}/submission_state_${RECIPE_SLUG}_${CORPUS_NAME}_${RUN_STAMP}}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH}/formosan_mt_experiments/manifests}"

TRAIN_SL="${TRAIN_SL:-${EXP_DIR}/slurm/train_directional.sl}"
EVAL_SL="${EVAL_SL:-${EXP_DIR}/slurm/evaluate_directional.sl}"
VALIDATE_SL="${VALIDATE_SL:-${EXP_DIR}/slurm/validate_corpus.sl}"
NLLB_SETUP_SL="${NLLB_SETUP_SL:-${EXP_DIR}/slurm/setup_spm_sweep.sl}"
MADLAD_SETUP_SL="${MADLAD_SETUP_SL:-${EXP_DIR}/slurm/setup_madlad400.sl}"
NLLB_SETUP_IMPLEMENTATION="${NLLB_SETUP_IMPLEMENTATION:-${EXP_DIR}/scripts/setup_formosan_nllb200.py}"

mkdir -p "${STATE_DIR}" "${LOGS_DIR}"
for short in en zh; do
  input="${PROJECT_DATA}/${CORPUS_NAME}/big_corpus_${short}_in_domain_hard.csv"
  [[ -r "${input}" ]] || { echo "Missing corpus input: ${input}" >&2; exit 1; }
done
CORPUS_MANIFEST="${PROJECT_DATA}/${CORPUS_NAME}/provenance/mt_build_manifest.json"
[[ -r "${CORPUS_MANIFEST}" ]] || {
  echo "Missing corpus provenance: ${CORPUS_MANIFEST}" >&2
  exit 1
}
[[ -r "${PROFILE}" ]] || { echo "Missing profile: ${PROFILE}" >&2; exit 1; }

job_state() {
  local id="$1"
  local state=""
  local attempt
  for attempt in {1..10}; do
    state="$(sacct -j "${id}" --starttime 2020-01-01 --noheader \
      --parsable2 --allocations --format=State 2>/dev/null \
      | awk -F'|' 'NF && $1 != "" {print $1; exit}')"
    if [[ -n "${state}" ]]; then
      printf '%s' "${state}"
      return 0
    fi
    sleep 1
  done
}

submit_job() {
  local label="$1"
  shift
  local record="${STATE_DIR}/${label}.id"
  local out state
  if [[ -s "${record}" ]]; then
    out="$(<"${record}")"
    state="$(job_state "${out}")"
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
        echo "Cannot classify recorded job ${out} for ${label}: ${state:-unknown}" >&2
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
  cat "${STATE_DIR}/$1.id"
}

dependency_for() {
  local ids=()
  local label id state
  for label in "$@"; do
    [[ -s "${STATE_DIR}/${label}.id" ]] || continue
    id="$(job_id "${label}")"
    state="$(job_state "${id}")"
    case "${state}" in
      COMPLETED*) ;;
      PENDING*|RUNNING*|CONFIGURING*|COMPLETING*) ids+=("${id}") ;;
      *)
        echo "Cannot depend on ${label}=${id}; state=${state:-unknown}" >&2
        exit 1
        ;;
    esac
  done
  if ((${#ids[@]})); then
    local joined
    joined="$(IFS=:; echo "${ids[*]}")"
    printf '%s' "--dependency=afterok:${joined}"
  fi
}

common_export() {
  local target_lang="$1"
  local direction="${2:-}"
  local pieces=(
    "EXP_DIR=${EXP_DIR}"
    "SCRATCH=${SCRATCH}"
    "PROJECT_DATA=${PROJECT_DATA}"
    "CORPUS_NAME=${CORPUS_NAME}"
    "TARGET_LANG=${target_lang}"
    "PROFILE=${PROFILE}"
  )
  [[ -z "${direction}" ]] || pieces+=("DIRECTION=${direction}")
  local joined
  joined="$(IFS=,; echo "${pieces[*]}")"
  printf 'ALL,%s' "${joined}"
}

submit_validation() {
  local target_lang="$1"
  local short="$2"
  submit_job "validate_${short}" \
    --job-name="${MODEL_FAMILY}_${CORPUS_NAME}_${short}_validate" \
    --partition="${VALIDATE_PARTITION:-short}" \
    --time="${VALIDATE_TIME:-02:00:00}" \
    --cpus-per-task="${VALIDATE_CPUS:-8}" \
    --mem="${VALIDATE_MEM:-64G}" \
    --export="$(common_export "${target_lang}"),MIN_TEST_RATIO=0.075,MIN_VALIDATE_RATIO=0.025" \
    "${VALIDATE_SL}"
}

nllb_setup_paths() {
  local short="$1"
  local root="${DATA_DIR}/tokenizer_sweep_${short}_spm8192"
  printf '%s|%s|%s' \
    "${root}/formosan_multilingual_nllb_spm8192_tokenizer" \
    "${root}/formosan_multilingual_nllb_spm8192_model" \
    "${root}/formosan_multilingual_nllb_spm8192_setup_manifest.json"
}

madlad_setup_paths() {
  local root="${DATA_DIR}/${RECIPE_SLUG}/setup"
  printf '%s|%s|%s' \
    "${root}/tokenizer" \
    "${root}/model" \
    "${root}/setup_manifest.json"
}

setup_complete() {
  local tokenizer="$1"
  local model="$2"
  local manifest="$3"
  shift 3
  [[ -f "${tokenizer}/tokenizer_config.json" \
    && -f "${model}/config.json" \
    && -f "${manifest}" ]] || return 1
  compgen -G "${model}/model*.safetensors" > /dev/null || return 1
  python - "${manifest}" "${PROFILE}" "${MODEL_FAMILY}" "$@" <<'PY'
import hashlib
import json
import sys

manifest_path, profile_path, family, *inputs = sys.argv[1:]
value = json.load(open(manifest_path, encoding="utf-8"))
profile_hash = hashlib.sha256(open(profile_path, "rb").read()).hexdigest()
input_hashes = {
    hashlib.sha256(open(path, "rb").read()).hexdigest()
    for path in inputs
}
records = value.get("inputs")
if not isinstance(records, list):
    records = [value.get("input", {})]
recorded_hashes = {
    record.get("sha256")
    for record in records
    if isinstance(record, dict)
}
actual_family = str(value.get("model_family") or "nllb")
ok = (
    value.get("complete") is True
    and actual_family == family
    and value.get("profile", {}).get("sha256") == profile_hash
    and input_hashes <= recorded_hashes
)
raise SystemExit(0 if ok else 1)
PY
}

submit_nllb_setup() {
  local target_lang="$1"
  local short="$2"
  local paths tokenizer model manifest root dependency setup_sha
  paths="$(nllb_setup_paths "${short}")"
  IFS='|' read -r tokenizer model manifest <<<"${paths}"
  root="$(dirname "${tokenizer}")"
  local input="${PROJECT_DATA}/${CORPUS_NAME}/big_corpus_${short}_in_domain_hard.csv"
  if setup_complete "${tokenizer}" "${model}" "${manifest}" "${input}"; then
    echo "setup_${short}_spm8192=already_exists ${root}"
    return
  fi
  dependency="$(dependency_for "validate_${short}")"
  setup_sha="$(sha256sum "${NLLB_SETUP_IMPLEMENTATION}" | awk '{print $1}')"
  local args=(
    --job-name="nllb_${CORPUS_NAME}_${short}_setup"
    --partition="${SETUP_PARTITION:-short}"
    --time="${SETUP_TIME:-12:00:00}"
    --cpus-per-task="${SETUP_CPUS:-16}"
    --mem="${SETUP_MEM:-128G}"
  )
  [[ -z "${dependency}" ]] || args+=("${dependency}")
  args+=(
    --export="$(common_export "${target_lang}"),OUT_DIR=${root},SPM_VOCABS=8192,SETUP_SCRIPT=${NLLB_SETUP_IMPLEMENTATION},SETUP_SCRIPT_SHA256=${setup_sha}"
    "${NLLB_SETUP_SL}"
  )
  submit_job "setup_${short}_spm8192" "${args[@]}"
}

submit_madlad_setup() {
  local paths tokenizer model manifest root dependency
  paths="$(madlad_setup_paths)"
  IFS='|' read -r tokenizer model manifest <<<"${paths}"
  root="$(dirname "${tokenizer}")"
  if setup_complete \
    "${tokenizer}" \
    "${model}" \
    "${manifest}" \
    "${PROJECT_DATA}/${CORPUS_NAME}/big_corpus_en_in_domain_hard.csv" \
    "${PROJECT_DATA}/${CORPUS_NAME}/big_corpus_zh_in_domain_hard.csv"; then
    echo "setup_madlad400=already_exists ${root}"
    return
  fi
  dependency="$(dependency_for validate_en validate_zh)"
  local args=(
    --job-name="madlad_${CORPUS_NAME}_setup"
    --partition="${SETUP_PARTITION:-short}"
    --time="${SETUP_TIME:-12:00:00}"
    --cpus-per-task="${SETUP_CPUS:-16}"
    --mem="${SETUP_MEM:-128G}"
  )
  [[ -z "${dependency}" ]] || args+=("${dependency}")
  args+=(
    --export="$(common_export english),OUT_DIR=${root}"
    "${MADLAD_SETUP_SL}"
  )
  submit_job setup_madlad400 "${args[@]}"
}

submit_direction() {
  local target_lang="$1"
  local short="$2"
  local direction="$3"
  local paths tokenizer model setup_manifest setup_label
  if [[ "${MODEL_FAMILY}" == "nllb" ]]; then
    paths="$(nllb_setup_paths "${short}")"
    setup_label="setup_${short}_spm8192"
  else
    paths="$(madlad_setup_paths)"
    setup_label="setup_madlad400"
  fi
  IFS='|' read -r tokenizer model setup_manifest <<<"${paths}"

  local validation_report="${PROJECT_DATA}/${CORPUS_NAME}/provenance/validate_${short}_in_domain_hard_runtime.json"
  local run_out="${RUNS_DIR}/${RECIPE_SLUG}_${direction}_${RUN_STAMP}"
  local dependency
  dependency="$(dependency_for "validate_${short}" "${setup_label}")"

  local default_constraint default_mem default_time default_eval_batch
  if [[ "${MODEL_FAMILY}" == "madlad400" ]]; then
    default_constraint="vr80g|vr144g"
    default_mem="160G"
    default_time="2-00:00:00"
    default_eval_batch="1"
  else
    default_constraint="vr40g|vr80g|vr144g"
    default_mem="128G"
    default_time="2-00:00:00"
    default_eval_batch="16"
  fi
  local train_args=(
    --job-name="${MODEL_FAMILY}_${CORPUS_NAME}_${direction}"
    --partition="${TRAIN_PARTITION:-medium}"
    --time="${TRAIN_TIME:-${default_time}}"
    --gres="${TRAIN_GRES:-gpu:1}"
    --constraint="${TRAIN_CONSTRAINT:-${default_constraint}}"
    --cpus-per-task="${TRAIN_CPUS:-8}"
    --mem="${TRAIN_MEM:-${default_mem}}"
  )
  [[ -z "${dependency}" ]] || train_args+=("${dependency}")
  train_args+=(
    --export="$(common_export "${target_lang}" "${direction}"),TOKENIZER=${tokenizer},MODEL=${model},SETUP_MANIFEST=${setup_manifest},CORPUS_MANIFEST=${CORPUS_MANIFEST},VALIDATION_REPORT=${validation_report},OUT_DIR=${run_out}"
    "${TRAIN_SL}"
  )
  submit_job "train_${direction}" "${train_args[@]}"

  local train_id train_state
  train_id="$(job_id "train_${direction}")"
  train_state="$(job_state "${train_id}")"
  local eval_dependency=""
  case "${train_state}" in
    COMPLETED*) ;;
    PENDING*|RUNNING*|CONFIGURING*|COMPLETING*)
      eval_dependency="--dependency=afterok:${train_id}"
      ;;
    *)
      echo "Cannot submit evaluations for ${train_id}; state=${train_state:-unknown}" >&2
      exit 1
      ;;
  esac
  local checkpoint
  for checkpoint in final best; do
    local eval_args=(
      --job-name="${MODEL_FAMILY}_${CORPUS_NAME}_${direction}_eval_${checkpoint}"
      --partition="${EVAL_PARTITION:-medium}"
      --time="${EVAL_TIME:-1-00:00:00}"
      --gres="${EVAL_GRES:-gpu:1}"
      --constraint="${EVAL_CONSTRAINT:-${default_constraint}}"
      --cpus-per-task="${EVAL_CPUS:-8}"
      --mem="${EVAL_MEM:-${default_mem}}"
    )
    [[ -z "${eval_dependency}" ]] || eval_args+=("${eval_dependency}")
    eval_args+=(
      --export="$(common_export "${target_lang}" "${direction}"),MODEL=${run_out}/${checkpoint},TOKENIZER=${run_out}/${checkpoint},CORPUS_MANIFEST=${CORPUS_MANIFEST},VALIDATION_REPORT=${validation_report},RUN_CONTRACT=${run_out}/run_contract.json,OUT_DIR=${REPORTS_DIR}/${RECIPE_SLUG}_${direction}_${checkpoint}_${RUN_STAMP},BATCH_SIZE=${default_eval_batch}"
      "${EVAL_SL}"
    )
    submit_job "eval_${direction}_${checkpoint}" "${eval_args[@]}"
  done
}

echo "RUN_STAMP=${RUN_STAMP}"
echo "CORPUS_NAME=${CORPUS_NAME}"
echo "MODEL_FAMILY=${MODEL_FAMILY}"
echo "RECIPE_ID=${RECIPE_ID}"
echo "STATE_DIR=${STATE_DIR}"
echo "LOGS_DIR=${LOGS_DIR}"

submit_validation english en
submit_validation chinese zh
if [[ "${MODEL_FAMILY}" == "nllb" ]]; then
  submit_nllb_setup english en
  submit_nllb_setup chinese zh
else
  submit_madlad_setup
fi

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
  --output "${MANIFEST_DIR}/submission_manifest_${RECIPE_SLUG}_${CORPUS_NAME}_${RUN_STAMP}.json"
touch "${STATE_DIR}/DONE"
echo "DONE_SUBMIT RUN_STAMP=${RUN_STAMP} RECIPE_ID=${RECIPE_ID}"
