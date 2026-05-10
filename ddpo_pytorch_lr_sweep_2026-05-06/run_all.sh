#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

bash "${HERE}/run_one.sh" "${HERE}/configs/lr_1e_4.py" "lr_1e_4"
bash "${HERE}/run_one.sh" "${HERE}/configs/lr_6e_4.py" "lr_6e_4"
bash "${HERE}/run_one.sh" "${HERE}/configs/clip_5e_4.py" "clip_5e_4"
