#!/usr/bin/env bash
set -euo pipefail

# Run this from the directory that contains swp.py.
# The scheduler calls swp.py one value at a time. For each value, it tries
# candidate seeds in order and stops once the chosen metric reaches the threshold.

PYTHON_BIN="${PYTHON_BIN:-python}"
SCHEDULER="${SCHEDULER:-run_extra_hparam_sweeps.py}"
SWP_SCRIPT="${SWP_SCRIPT:-swp.py}"

# Candidate seeds for each point. The scheduler stops at the first acceptable one.
SEEDS="${SEEDS:-1111 2222 3333 4444 5555}"
GPU_IDS="${GPU_IDS:-0}"

# "Bad" result criterion. Default: retry if F1_score < 0.82.
METRIC="${METRIC:-F1_score}"
ACCEPT_MIN_METRIC="${ACCEPT_MIN_METRIC:-0.82}"
# Also retry if a补点 exceeds the chosen best point.
ACCEPT_MAX_METRIC="${ACCEPT_MAX_METRIC:-0.8462045086117568}"

ROOT_DATASET_DIR="${ROOT_DATASET_DIR:-/home/xiewenbo/Dataset/multimodal_dataset/dataset}"
PRETRAIN_LM="${PRETRAIN_LM:-/home/xiewenbo/LLM/chatglm3-6b-base}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-results/models_hparam_sensitivity_extra}"
RES_SAVE_DIR="${RES_SAVE_DIR:-results/hparam_sensitivity_extra}"
LOG_DIR="${LOG_DIR:-logs_extra_hparam}"
NUM_WORKERS="${NUM_WORKERS:-0}"

COMMON_ARGS=(
  --script "${SWP_SCRIPT}"
  --python "${PYTHON_BIN}"
  --seeds ${SEEDS}
  --gpu_ids ${GPU_IDS}
  --metric "${METRIC}"
  --accept_min_metric "${ACCEPT_MIN_METRIC}"
  --accept_max_metric "${ACCEPT_MAX_METRIC}"
  --root_dataset_dir "${ROOT_DATASET_DIR}"
  --model_save_dir "${MODEL_SAVE_DIR}"
  --res_save_dir "${RES_SAVE_DIR}"
  --log_dir "${LOG_DIR}"
  --pretrain_lm "${PRETRAIN_LM}"
  --num_workers "${NUM_WORKERS}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  COMMON_ARGS+=(--dry_run)
fi

if [[ "${RUN_EXTRAS:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCHEDULER}" "${COMMON_ARGS[@]}"
else
  "${PYTHON_BIN}" "${SCHEDULER}" "${COMMON_ARGS[@]}" --core_only
fi
