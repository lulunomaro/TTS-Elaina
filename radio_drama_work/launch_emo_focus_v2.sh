#!/bin/bash
set -euo pipefail

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
TRAIN=/root/autodl-tmp/s2_train_dlr.py
EXP=/root/autodl-tmp/exp/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5
LOG=/root/autodl-tmp/train_ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5.log

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY" "$TRAIN" -c "$EXP/cfg/s2.json" >> "$LOG" 2>&1
