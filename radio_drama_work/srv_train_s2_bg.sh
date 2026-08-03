#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: srv_train_s2_bg.sh EXP_NAME" >&2
  exit 2
fi

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=$1
EXP_DIR=/root/autodl-tmp/exp/$EXP_NAME
CFG=$EXP_DIR/cfg
LOG=/root/autodl-tmp/train_${EXP_NAME}.log

if pgrep -af "s2_train.py" >/dev/null 2>&1; then
  echo "an s2_train.py process is already running; refusing to start another one" >&2
  exit 3
fi

if [ ! -f "$CFG/s2.json" ] || [ ! -f "$EXP_DIR/6-name2semantic.tsv" ]; then
  echo "missing config or prepared semantic data under $EXP_DIR" >&2
  exit 4
fi

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup setsid "$PY" GPT_SoVITS/s2_train.py -c "$CFG/s2.json" > "$LOG" 2>&1 < /dev/null &
disown || true
echo "launched pid=$! log=$LOG"
