#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/inspire/hdd/project/robot-reasoning/czxs253130583}"
OPENPI_DIR="${OPENPI_DIR:-${USER_ROOT}/openpi_ws/openpi}"
OPENPI_ENV="${USER_ROOT}/openpi_env"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ENV}/bin/python}"
DATASET_DIR="${DATASET_DIR:-${USER_ROOT}/r1lite_datasets/lerobot_20261022_current3cam}"
EXP_NAME="${EXP_NAME:-r1lite-full-lora}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
OVERWRITE="${OVERWRITE:-0}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${USER_ROOT}/openpi_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${USER_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/hf}"
export OPENPI_LOCAL_DATASET_ROOT="${OPENPI_LOCAL_DATASET_ROOT:-${DATASET_DIR}}"

mkdir -p "${OPENPI_DATA_HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}"
cd "${OPENPI_DIR}"

TRAIN_FLAGS=()
CHECKPOINT_DIR="${USER_ROOT}/openpi_r1lite_checkpoints/pi0_r1lite/${EXP_NAME}"
if [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_FLAGS+=(--overwrite)
elif [[ -n "$(find "${CHECKPOINT_DIR}" -type f -print -quit 2>/dev/null)" ]]; then
  TRAIN_FLAGS+=(--resume)
fi

exec "${PYTHON_BIN}" scripts/train.py pi0_r1lite \
  --exp-name "${EXP_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  "${TRAIN_FLAGS[@]}" "$@"
