#!/bin/bash
set -euo pipefail
REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP=ireina_s2_balanced48_plus18_v1
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PY" GPT_SoVITS/s2_train.py -c "/root/autodl-tmp/exp/$EXP/cfg/s2.json"
