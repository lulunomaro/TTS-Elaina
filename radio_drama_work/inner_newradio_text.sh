#!/bin/bash
set -u
LOG=/root/autodl-tmp/train_ireina_s2_newradio_text_lr15_b32_v2.log
exec >>"$LOG" 2>&1
echo "START $(date)"
REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
TRAIN=/root/autodl-tmp/s2_train_dlr.py
EXP=/root/autodl-tmp/exp/ireina_s2_newradio_text_lr15_b32_v2
cd "$REPO" || exit 10
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PY" "$TRAIN" -c "$EXP/cfg/s2.json"
