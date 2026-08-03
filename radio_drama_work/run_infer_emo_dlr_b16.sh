#!/bin/bash
set -euo pipefail
export ONLY_NEW_EMO20_DLR2E5_B16=1
exec /root/miniconda3/envs/gptsovits/bin/python /
  /root/autodl-tmp/srv_infer_refine_v3.py /
  /root/autodl-tmp/infer_balanced48_emo20_dlr2e5_b16_v1 /
  > /root/autodl-tmp/infer_balanced48_emo20_dlr2e5_b16_v1.log 2>&1
