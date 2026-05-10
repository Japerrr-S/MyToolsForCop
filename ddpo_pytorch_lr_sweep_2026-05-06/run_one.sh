#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:?Usage: run_one.sh CONFIG_PATH RUN_TAG}
RUN_TAG=${2:?Usage: run_one.sh CONFIG_PATH RUN_TAG}

OUT_DIR=$(cd "$(dirname "$0")" && pwd)
DDPO_DATA_DIR=${DDPO_DATA_DIR:-/root/autodl-tmp}
DDPO_PYTORCH_ROOT=${DDPO_PYTORCH_ROOT:-/root/autodl-tmp/ddpo-pytorch}

export XDG_CACHE_HOME="${DDPO_DATA_DIR}/.cache"
export HF_HOME="${DDPO_DATA_DIR}/.cache/huggingface"
export DIFFUSERS_CACHE="${DDPO_DATA_DIR}/.cache/huggingface/diffusers"
export TORCH_HOME="${DDPO_DATA_DIR}/.cache/torch"
export PIP_CACHE_DIR="${DDPO_DATA_DIR}/.cache/pip"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"${OUT_DIR}/wandb_offline"}
export WANDB_SILENT=${WANDB_SILENT:-true}
export PYTHONPATH="${DDPO_PYTORCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${OUT_DIR}/stdout"

if command -v readlink >/dev/null 2>&1; then
    CONFIG_PATH=$(readlink -f "${CONFIG_PATH}" || echo "${CONFIG_PATH}")
fi

LOG_FILE="${OUT_DIR}/stdout/${RUN_TAG}.log"

echo "=== CONFIG: ${CONFIG_PATH} ==="
echo "=== RUN_TAG: ${RUN_TAG} ==="
echo "=== LOG_FILE: ${LOG_FILE} ==="
echo "=== HF_ENDPOINT: ${HF_ENDPOINT} ==="

cd "${DDPO_PYTORCH_ROOT}"
stdbuf -oL -eL accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    scripts/train.py --config "${CONFIG_PATH}" \
    2>&1 | tee "${LOG_FILE}"
