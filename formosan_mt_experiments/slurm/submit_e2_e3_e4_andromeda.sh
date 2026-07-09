#!/bin/bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
EXP_DIR="${EXP_DIR:-/home/scheppat/workspace/projects/mt/formosan_mt_experiments}"
SCRATCH="${SCRATCH:-/scratch/scheppat}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/formosan_mt_experiments/data}"
SPLIT_DIR="${SPLIT_DIR:-/projects/prudlab/formosan_parallel_corpora/splits_en_v1}"
TOKEN_DIR="${TOKEN_DIR:-${DATA_DIR}/tokenizer_sweep}"
TOKEN_16="${TOKEN_16:-${TOKEN_DIR}/formosan_multilingual_nllb_spm16384_tokenizer}"
MODEL_16="${MODEL_16:-${TOKEN_DIR}/formosan_multilingual_nllb_spm16384_model}"
INPUT="${INPUT:-${SPLIT_DIR}/big_corpus_en_in_domain_hard.csv}"
JOBS_DIR="${JOBS_DIR:-/home/scheppat/jobs/mt}"
STATE_DIR="${STATE_DIR:-${JOBS_DIR}/submission_state_${RUN_STAMP}}"

TRAIN_SL="${TRAIN_SL:-${JOBS_DIR}/formosan_exp_train_directional.sl}"
EVAL_SL="${EVAL_SL:-${JOBS_DIR}/formosan_exp_evaluate_directional.sl}"
DAE_SL="${DAE_SL:-${JOBS_DIR}/formosan_exp_pretrain_dae.sl}"
SETUP_SL="${SETUP_SL:-${JOBS_DIR}/formosan_exp_setup_spm_sweep.sl}"
LEGACY_SL="${LEGACY_SL:-${JOBS_DIR}/formosan_exp_train_legacy_multilingual.sl}"
LEGACY_TOKENIZER="${LEGACY_TOKENIZER:-${SCRATCH}/nllb200-multilingual/tokenizer/formosan_multilingual_nllb_tokenizer}"
LEGACY_MODEL="${LEGACY_MODEL:-${SCRATCH}/nllb200-multilingual/model/formosan_multilingual_nllb_model}"

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

# E0: old multilingual scripts + old char-added NLLB tokenizer/model on the new
# in_domain_hard split. This isolates "harder data" from E1-E4 method changes.
E0_OUT="${SCRATCH}/formosan_mt_experiments/runs/E0_legacy_multilingual_bidir_${RUN_STAMP}"
submit_job E0_legacy_multilingual_train \
  --job-name=E0_legacy_multilingual \
  --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=96G \
  --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",RUN_STAMP="${RUN_STAMP}",TGT_LANG=english,INPUT="${INPUT}",TOKENIZER="${LEGACY_TOKENIZER}",MODEL="${LEGACY_MODEL}",OUT_DIR="${E0_OUT}",STEPS=300000,BATCH_SIZE=16,GRAD_ACCUM_STEPS=4,MAX_LENGTH=256,LEARNING_RATE=3e-5,SAVE_INTERVAL=25000,EVAL_INTERVAL=25000,EVAL_SAMPLE_SIZE=256,EVAL_BATCH_SIZE=16,P_SRC2TGT=0.5,EASY_SOURCE_WEIGHT=0.1,ALPHA=0.5,NORMALIZE=1 \
  "${LEGACY_SL}"

# E2: DAE pre-adaptation on the 16k SPM 600M model, then directional MT from DAE best.
E2_DAE_OUT="${SCRATCH}/formosan_mt_experiments/runs/E2_dae_spm16k_${RUN_STAMP}"
submit_job E2_DAE \
  --job-name=E2_dae_spm16k \
  --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=128G \
  --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,INPUT="${INPUT}",TOKENIZER="${TOKEN_16}",MODEL="${MODEL_16}",OUT_DIR="${E2_DAE_OUT}",STEPS=100000,BATCH_SIZE=16,GRAD_ACCUM_STEPS=4,MAX_LENGTH=384,LEARNING_RATE=3e-5,SAVE_INTERVAL=25000,EVAL_INTERVAL=25000,EVAL_SAMPLES=256,EVAL_BATCH_SIZE=16 \
  "${DAE_SL}"
E2_DAE_ID="$(job_id E2_DAE)"

for DIRN in f2en en2f; do
  RUN_OUT="${SCRATCH}/formosan_mt_experiments/runs/E2_dae_spm16k_${DIRN}_${RUN_STAMP}"
  TRAIN_LABEL="E2_${DIRN}_train"
  submit_job "${TRAIN_LABEL}" \
    --job-name="E2_${DIRN}_dae" \
    --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=128G \
    --dependency="afterok:${E2_DAE_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",TOKENIZER="${TOKEN_16}",USE_DAE=1,DAE_MODEL="${E2_DAE_OUT}/best",OUT_DIR="${RUN_OUT}",STEPS=300000,BATCH_SIZE=16,GRAD_ACCUM_STEPS=4,MAX_LENGTH=384,LEARNING_RATE=3e-5,SAVE_INTERVAL=25000,EVAL_INTERVAL=25000,EVAL_SAMPLES=256,EVAL_BATCH_SIZE=16 \
    "${TRAIN_SL}"
  TRAIN_ID="$(job_id "${TRAIN_LABEL}")"
  submit_job "E2_${DIRN}_eval_final" \
    --job-name="E2_${DIRN}_eval_final" \
    --partition=medium --time=08:00:00 --gres=gpu:h200:1 --mem=96G \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/final",TOKENIZER="${RUN_OUT}/final",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E2_dae_spm16k_${DIRN}_final_${RUN_STAMP}",BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
    "${EVAL_SL}"
  submit_job "E2_${DIRN}_eval_best" \
    --job-name="E2_${DIRN}_eval_best" \
    --partition=medium --time=08:00:00 --gres=gpu:h200:1 --mem=96G \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/best",TOKENIZER="${RUN_OUT}/best",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E2_dae_spm16k_${DIRN}_best_${RUN_STAMP}",BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
    "${EVAL_SL}"
done

# E3: SPM vocab sweep. E1 supplies the 16k point; submit 8k and 32k setup+directional training.
for VOCAB in 8192 32768; do
  SETUP_OUT="${DATA_DIR}/tokenizer_sweep_spm${VOCAB}"
  submit_job "E3_setup_${VOCAB}" \
    --job-name="E3_setup_spm${VOCAB}" \
    --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=128G \
    --export=ALL,EXP_DIR="${EXP_DIR}",INPUT="${INPUT}",OUT_DIR="${SETUP_OUT}",SETUP_SCRIPT=/home/scheppat/nllb-scripts/setup_formosan_nllb200.py,SPM_VOCABS="${VOCAB}",SETUP_SPLITS=train,validate,BASE_MODEL=facebook/nllb-200-distilled-600M \
    "${SETUP_SL}"
  SETUP_ID="$(job_id "E3_setup_${VOCAB}")"
  TOK="${SETUP_OUT}/formosan_multilingual_nllb_spm${VOCAB}_tokenizer"
  MOD="${SETUP_OUT}/formosan_multilingual_nllb_spm${VOCAB}_model"
  for DIRN in f2en en2f; do
    RUN_OUT="${SCRATCH}/formosan_mt_experiments/runs/E3_spm${VOCAB}_${DIRN}_${RUN_STAMP}"
    TRAIN_LABEL="E3_${VOCAB}_${DIRN}_train"
    submit_job "${TRAIN_LABEL}" \
      --job-name="E3_${VOCAB}_${DIRN}" \
      --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=128G \
      --dependency="afterok:${SETUP_ID}" \
      --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",TOKENIZER="${TOK}",MODEL="${MOD}",OUT_DIR="${RUN_OUT}",STEPS=300000,BATCH_SIZE=16,GRAD_ACCUM_STEPS=4,MAX_LENGTH=384,LEARNING_RATE=2e-5,SAVE_INTERVAL=25000,EVAL_INTERVAL=25000,EVAL_SAMPLES=256,EVAL_BATCH_SIZE=16 \
      "${TRAIN_SL}"
    TRAIN_ID="$(job_id "${TRAIN_LABEL}")"
    submit_job "E3_${VOCAB}_${DIRN}_eval_final" \
      --job-name="E3_${VOCAB}_${DIRN}_eval_final" \
      --partition=medium --time=08:00:00 --gres=gpu:h200:1 --mem=96G \
      --dependency="afterok:${TRAIN_ID}" \
      --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/final",TOKENIZER="${RUN_OUT}/final",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E3_spm${VOCAB}_${DIRN}_final_${RUN_STAMP}",BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
      "${EVAL_SL}"
    submit_job "E3_${VOCAB}_${DIRN}_eval_best" \
      --job-name="E3_${VOCAB}_${DIRN}_eval_best" \
      --partition=medium --time=08:00:00 --gres=gpu:h200:1 --mem=96G \
      --dependency="afterok:${TRAIN_ID}" \
      --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/best",TOKENIZER="${RUN_OUT}/best",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E3_spm${VOCAB}_${DIRN}_best_${RUN_STAMP}",BATCH_SIZE=16,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
      "${EVAL_SL}"
  done
done

# E4: NLLB-1.3B + LoRA/PEFT, custom 16k SPM tokenizer/model built from the 1.3B base.
E4_SETUP_OUT="${DATA_DIR}/tokenizer_sweep_1p3b"
submit_job E4_setup_1p3b_spm16k \
  --job-name=E4_setup_1p3b_spm16k \
  --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=192G \
  --export=ALL,EXP_DIR="${EXP_DIR}",INPUT="${INPUT}",OUT_DIR="${E4_SETUP_OUT}",SETUP_SCRIPT=/home/scheppat/nllb-scripts/setup_formosan_nllb200.py,SPM_VOCABS=16384,SETUP_SPLITS=train,validate,BASE_MODEL=facebook/nllb-200-1.3B \
  "${SETUP_SL}"
E4_SETUP_ID="$(job_id E4_setup_1p3b_spm16k)"
E4_TOKEN="${E4_SETUP_OUT}/formosan_multilingual_nllb_spm16384_tokenizer"
E4_MODEL="${E4_SETUP_OUT}/formosan_multilingual_nllb_spm16384_model"
for DIRN in f2en en2f; do
  RUN_OUT="${SCRATCH}/formosan_mt_experiments/runs/E4_1p3b_lora_spm16k_${DIRN}_${RUN_STAMP}"
  TRAIN_LABEL="E4_${DIRN}_train"
  submit_job "${TRAIN_LABEL}" \
    --job-name="E4_1p3b_lora_${DIRN}" \
    --partition=medium --time=2-00:00:00 --gres=gpu:h200:1 --mem=192G \
    --dependency="afterok:${E4_SETUP_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",TOKENIZER="${E4_TOKEN}",MODEL="${E4_MODEL}",OUT_DIR="${RUN_OUT}",STEPS=150000,BATCH_SIZE=8,GRAD_ACCUM_STEPS=8,MAX_LENGTH=384,LEARNING_RATE=5e-5,SAVE_INTERVAL=25000,EVAL_INTERVAL=25000,EVAL_SAMPLES=256,EVAL_BATCH_SIZE=8,LORA_R=16,LORA_ALPHA=32,LORA_DROPOUT=0.05 \
    "${TRAIN_SL}"
  TRAIN_ID="$(job_id "${TRAIN_LABEL}")"
  submit_job "E4_${DIRN}_eval_final" \
    --job-name="E4_${DIRN}_eval_final" \
    --partition=medium --time=08:00:00 --gres=gpu:h200:1 --mem=128G \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/final",TOKENIZER="${RUN_OUT}/final",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E4_1p3b_lora_spm16k_${DIRN}_final_${RUN_STAMP}",BATCH_SIZE=8,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
    "${EVAL_SL}"
  submit_job "E4_${DIRN}_eval_best" \
    --job-name="E4_${DIRN}_eval_best" \
    --partition=medium --time=08:00:00 --gres=gpu:h200:1 --mem=128G \
    --dependency="afterok:${TRAIN_ID}" \
    --export=ALL,EXP_DIR="${EXP_DIR}",SCRATCH="${SCRATCH}",TIER=in_domain_hard,DIRECTION="${DIRN}",INPUT="${INPUT}",MODEL="${RUN_OUT}/best",TOKENIZER="${RUN_OUT}/best",OUT_DIR="${SCRATCH}/formosan_mt_experiments/reports/E4_1p3b_lora_spm16k_${DIRN}_best_${RUN_STAMP}",BATCH_SIZE=8,MAX_LENGTH=384,BEAM=4,MAX_NEW_TOKENS=256 \
    "${EVAL_SL}"
done

touch "${STATE_DIR}/DONE"
echo "DONE_SUBMIT RUN_STAMP=${RUN_STAMP}"
