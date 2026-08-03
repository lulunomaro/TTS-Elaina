#!/bin/bash
# AutoDL 服务器端：v2Pro S1(GPT) + S2(SoVITS) 训练
set -e

REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=${EXP_NAME:-ireina_v1}
EXP_DIR=/root/autodl-tmp/exp/$EXP_NAME
CFG=$EXP_DIR/cfg

cd $REPO
export PYTHONPATH=$REPO:$REPO/GPT_SoVITS:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0

echo "########## S1 (GPT) 训练开始 $(date +%H:%M:%S) ##########"
$PY GPT_SoVITS/s1_train.py --config_file $CFG/s1.yaml 2>&1 | tail -40
echo "S1 done $(date +%H:%M:%S)"
ls -la /root/autodl-tmp/weights/GPT_weights_v2Pro/ | tail -10

echo
echo "########## S2 (SoVITS) 训练开始 $(date +%H:%M:%S) ##########"
$PY GPT_SoVITS/s2_train.py -c $CFG/s2.json 2>&1 | tail -40
echo "S2 done $(date +%H:%M:%S)"
ls -la /root/autodl-tmp/weights/SoVITS_weights_v2Pro/ | tail -10

echo
echo "########## ALL TRAINING DONE ##########"
