#!/bin/bash
set -euo pipefail
REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
TRAIN=/root/autodl-tmp/s2_train_dlr.py
EXP=/root/autodl-tmp/exp/ireina_s2_newradio_gonly_lr5e6_b24_v2
LOG=/root/autodl-tmp/train_ireina_s2_newradio_gonly_lr5e6_b24_v2.log
SCREEN_NAME=ireina_newradio_gonly_v2
mkdir -p "$EXP/logs_s2_v2Pro"
screen -S "$SCREEN_NAME" -X quit >/dev/null 2>&1 || true
screen -dmS "$SCREEN_NAME" bash -lc "cd '$REPO'; export PYTHONPATH='$REPO:$REPO/GPT_SoVITS'; export CUDA_VISIBLE_DEVICES=0; export _CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec '$PY' '$TRAIN' -c '$EXP/cfg/s2.json' >> '$LOG' 2>&1"
echo "launched screen=$SCREEN_NAME log=$LOG"
