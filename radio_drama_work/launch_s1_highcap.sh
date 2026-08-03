#!/bin/bash
set -euo pipefail

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=${S1_EXP:?set S1_EXP}
EXP=/root/autodl-tmp/exp/$EXP_NAME
LOG=/root/autodl-tmp/train_${EXP_NAME}.log
SCREEN_NAME=${EXP_NAME}

screen -S "$SCREEN_NAME" -X quit >/dev/null 2>&1 || true
screen -dmS "$SCREEN_NAME" bash -lc "cd '$REPO'; export PYTHONPATH='$REPO:$REPO/GPT_SoVITS'; export CUDA_VISIBLE_DEVICES=0; export _CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec '$PY' GPT_SoVITS/s1_train.py --config_file '$EXP/cfg/s1.yaml' >> '$LOG' 2>&1"
echo "launched screen=$SCREEN_NAME log=$LOG"
