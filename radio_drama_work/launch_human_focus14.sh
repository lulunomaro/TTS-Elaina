#!/bin/bash
set -euo pipefail

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP=/root/autodl-tmp/exp/ireina_s2_human_focus14_lr5e5_v1
LOG=/root/autodl-tmp/train_ireina_s2_human_focus14_lr5e5_v1.log

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY" GPT_SoVITS/s2_train.py -c "$EXP/cfg/s2.json" >> "$LOG" 2>&1
