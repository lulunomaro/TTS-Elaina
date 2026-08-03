#!/bin/bash
set -euo pipefail
REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
LOG=/root/autodl-tmp/infer_textonly_v1.log
screen -S ireina_infer_textonly_v1 -X quit >/dev/null 2>&1 || true
screen -dmS ireina_infer_textonly_v1 bash -lc "cd '$REPO'; export PYTHONPATH='$REPO:$REPO/GPT_SoVITS'; export CUDA_VISIBLE_DEVICES=0; exec '$PY' /tmp/_wb_srv_infer_textonly_compare.py >> '$LOG' 2>&1"
echo "launched screen=ireina_infer_textonly_v1 log=$LOG"
