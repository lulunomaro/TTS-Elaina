#!/bin/bash
set -euo pipefail
EXP=ireina_s2_balanced48_plus18_v1
PY=/root/miniconda3/envs/gptsovits/bin/python
S2_EPOCHS=6 S2_SAVE_EVERY=2 "$PY" /root/autodl-tmp/srv_make_cfg_exp.py "$EXP"
bash /root/autodl-tmp/srv_train_s2_bg.sh "$EXP"
