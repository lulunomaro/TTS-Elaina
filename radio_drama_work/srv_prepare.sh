#!/bin/bash
# AutoDL 服务器端：v2Pro 数据预处理四件套
set -e

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=ireina_v1
EXP_DIR=/root/autodl-tmp/exp/$EXP_NAME
INP_TEXT=/root/autodl-tmp/dataset_v1/dataset.list

cd $REPO
mkdir -p $EXP_DIR

export inp_text=$INP_TEXT
export inp_wav_dir=""
export exp_name=$EXP_NAME
export opt_dir=$EXP_DIR
export is_half=True
export i_part=0
export all_parts=1
export _CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export version=v2Pro
export PYTHONPATH=$REPO:$REPO/GPT_SoVITS:$PYTHONPATH

export bert_pretrained_dir=$REPO/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
export cnhubert_base_dir=$REPO/GPT_SoVITS/pretrained_models/chinese-hubert-base
export sv_path=$REPO/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
export pretrained_s2G=$REPO/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
export s2config_path=$REPO/GPT_SoVITS/configs/s2v2Pro.json

echo "########## dataset check ##########"
wc -l $INP_TEXT
head -1 $INP_TEXT
echo

echo "########## 1-get-text ##########"
$PY GPT_SoVITS/prepare_datasets/1-get-text.py 2>&1 | tail -5
echo "   -> 2-name2text.txt: $(wc -l < $EXP_DIR/2-name2text.txt 2>/dev/null || echo 0) lines"

echo "########## 2-get-hubert-wav32k ##########"
$PY GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py 2>&1 | tail -5
echo "   -> 4-cnhubert: $(ls $EXP_DIR/4-cnhubert 2>/dev/null | wc -l) files"
echo "   -> 5-wav32k:   $(ls $EXP_DIR/5-wav32k 2>/dev/null | wc -l) files"

echo "########## 2-get-sv (v2Pro 专用) ##########"
$PY GPT_SoVITS/prepare_datasets/2-get-sv.py 2>&1 | tail -5
echo "   -> 7-sv_cn: $(ls $EXP_DIR/7-sv_cn 2>/dev/null | wc -l) files"

echo "########## 3-get-semantic ##########"
$PY GPT_SoVITS/prepare_datasets/3-get-semantic.py 2>&1 | tail -5
echo "   -> 6-name2semantic.tsv: $(wc -l < $EXP_DIR/6-name2semantic.tsv 2>/dev/null || echo 0) lines"

echo
echo "########## RESULT ##########"
ls -la $EXP_DIR
du -sh $EXP_DIR
