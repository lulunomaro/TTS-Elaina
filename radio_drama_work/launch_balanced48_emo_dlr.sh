#!/bin/bash
set -euo pipefail

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
TRAIN=/root/autodl-tmp/s2_train_dlr.py
EXP=/root/autodl-tmp/exp/ireina_s2_balanced48_emo20_dlr2e5_v1
LOG=/root/autodl-tmp/train_ireina_s2_balanced48_emo20_dlr2e5_v1.log

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY" "$TRAIN" -c "$EXP/cfg/s2.json" >> "$LOG" 2>&1
