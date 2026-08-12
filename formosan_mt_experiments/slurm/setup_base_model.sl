#!/bin/bash
#SBATCH --job-name=formosan_mt_base_setup
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load miniconda
conda activate formosan_mt

EXP_DIR="${EXP_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRATCH="${SCRATCH:-${HOME}/formosan_mt_work}"
PROFILE="${PROFILE:-${EXP_DIR}/configs/milmmt_1b_experiment.json}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for the pinned base-model snapshot}"
SETUP_SCRIPT="${SETUP_SCRIPT:-${EXP_DIR}/scripts/setup_milmmt.py}"
: "${SETUP_SCRIPT_SHA256:?SETUP_SCRIPT_SHA256 is required}"

export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${OUT_DIR}"

actual_setup_sha256="$(sha256sum "${SETUP_SCRIPT}" | awk '{print $1}')"
if [[ "${actual_setup_sha256}" != "${SETUP_SCRIPT_SHA256}" ]]; then
  echo "MiLMMT setup implementation checksum mismatch: ${actual_setup_sha256}" >&2
  exit 1
fi

srun --cpu-bind=cores python -u "${SETUP_SCRIPT}" \
  --profile "${PROFILE}" \
  --output-dir "${OUT_DIR}"
