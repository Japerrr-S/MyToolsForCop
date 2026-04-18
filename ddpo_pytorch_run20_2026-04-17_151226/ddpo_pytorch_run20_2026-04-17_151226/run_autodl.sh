#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-}
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: $0 <config.py> [run_tag]" >&2
  echo "Example: $0 config/autodl_4090_smoke.py smoke" >&2
  exit 2
fi

RUN_TAG=${2:-$(date +%F_%H%M%S)}

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Activate your conda env first (e.g. conda activate ddpo-pt)." >&2
  exit 2
fi

# Data dir (defaults to AutoDL data disk)
DDPO_DATA_DIR=${DDPO_DATA_DIR:-/root/autodl-tmp}

# Put all caches/logs on data disk to avoid filling system disk.
export XDG_CACHE_HOME="${DDPO_DATA_DIR}/.cache"
export HF_HOME="${DDPO_DATA_DIR}/.cache/huggingface"
export DIFFUSERS_CACHE="${DDPO_DATA_DIR}/.cache/huggingface/diffusers"
export TORCH_HOME="${DDPO_DATA_DIR}/.cache/torch"
export PIP_CACHE_DIR="${DDPO_DATA_DIR}/.cache/pip"

# HuggingFace endpoint (AutoDL/China networking often blocks huggingface.co)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# Increase read timeout for large file downloads (e.g. redirected to cas-bridge.xethub.hf.co)
export DDPO_HF_HTTP_TIMEOUT=${DDPO_HF_HTTP_TIMEOUT:-180}
export DDPO_HF_HTTP_MAX_RETRIES=${DDPO_HF_HTTP_MAX_RETRIES:-8}

# W&B offline logging
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"${DDPO_DATA_DIR}/wandb"}
export WANDB_SILENT=${WANDB_SILENT:-true}

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${DDPO_DATA_DIR}/ddpo-runs" \
  "${DDPO_DATA_DIR}/ddpo-pytorch-logs"

LOG_FILE="${DDPO_DATA_DIR}/ddpo-runs/${RUN_TAG}.log"

echo "=== CONFIG: ${CONFIG_PATH} ==="
echo "=== RUN_TAG: ${RUN_TAG} ==="
echo "=== LOG_FILE: ${LOG_FILE} ==="
echo "=== HF_ENDPOINT: ${HF_ENDPOINT} ==="
echo "=== DDPO_HF_HTTP_TIMEOUT: ${DDPO_HF_HTTP_TIMEOUT} ==="
echo "=== DDPO_HF_HTTP_MAX_RETRIES: ${DDPO_HF_HTTP_MAX_RETRIES} ==="

# Run from repo root
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_ROOT}"

stdbuf -oL -eL "${CONDA_PREFIX}/bin/accelerate" launch \
  --num_processes 1 \
  --mixed_precision fp16 \
  scripts/train.py --config "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_FILE}"
