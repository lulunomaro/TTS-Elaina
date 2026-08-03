#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: srv_prepare_exp.sh EXP_NAME DATASET_LIST" >&2
  exit 2
fi

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=$1
INP_TEXT=$2
EXP_DIR=/root/autodl-tmp/exp/$EXP_NAME

cd "$REPO"
mkdir -p "$EXP_DIR"

export inp_text="$INP_TEXT"
export inp_wav_dir=""
export exp_name="$EXP_NAME"
export opt_dir="$EXP_DIR"
export is_half=True
export i_part=0
export all_parts=1
export _CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export version=v2Pro
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS:${PYTHONPATH:-}"
export bert_pretrained_dir="$REPO/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="$REPO/GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="$REPO/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
export pretrained_s2G="$REPO/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
export s2config_path="$REPO/GPT_SoVITS/configs/s2v2Pro.json"

echo "dataset=$(wc -l < "$INP_TEXT") exp=$EXP_NAME"
"$PY" GPT_SoVITS/prepare_datasets/1-get-text.py
"$PY" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
"$PY" GPT_SoVITS/prepare_datasets/2-get-sv.py
"$PY" GPT_SoVITS/prepare_datasets/3-get-semantic.py
"$PY" /root/autodl-tmp/srv_merge_parts.py "$EXP_DIR"

echo "prepare done: $EXP_DIR"
du -sh "$EXP_DIR"
