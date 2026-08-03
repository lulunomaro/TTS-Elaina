#!/bin/bash
set -euo pipefail
/root/miniconda3/envs/gptsovits/bin/python /root/autodl-tmp/human_likeness_eval.py /root/autodl-tmp/human_likeness_eval_dold /root/autodl-tmp/ireina_v2/data/D_old.list
