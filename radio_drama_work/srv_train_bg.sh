#!/bin/bash
# 在服务器后台启动 S1+S2 训练（nohup，断开 SSH 也不影响）
REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=ireina_v1
EXP_DIR=/root/autodl-tmp/exp/$EXP_NAME
CFG=$EXP_DIR/cfg
LOG=/root/autodl-tmp/train_${EXP_NAME}.log

cat > /root/autodl-tmp/_run_train.sh <<'INNER'
set -e
REPO=/root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
EXP_NAME=ireina_v1
EXP_DIR=/root/autodl-tmp/exp/$EXP_NAME
CFG=$EXP_DIR/cfg
cd $REPO
export PYTHONPATH=$REPO:$REPO/GPT_SoVITS
export CUDA_VISIBLE_DEVICES=0
export _CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== S1 (GPT) START $(date +%F_%H:%M:%S) ====="
$PY GPT_SoVITS/s1_train.py --config_file $CFG/s1.yaml
echo "===== S1 DONE $(date +%F_%H:%M:%S) ====="
ls -la /root/autodl-tmp/weights/GPT_weights_v2Pro/

echo "===== S2 (SoVITS) START $(date +%F_%H:%M:%S) ====="
$PY GPT_SoVITS/s2_train.py -c $CFG/s2.json
echo "===== S2 DONE $(date +%F_%H:%M:%S) ====="
ls -la /root/autodl-tmp/weights/SoVITS_weights_v2Pro/

echo "===== ALL TRAINING DONE ====="
INNER

chmod +x /root/autodl-tmp/_run_train.sh

# 杀掉残留训练进程 + 清理上次的中断 ckpt（否则 lightning 会 resume 到坏状态）
pkill -f "s1_train.py" 2>/dev/null
pkill -f "s2_train.py" 2>/dev/null
sleep 5
rm -rf $EXP_DIR/logs_s1_v2Pro/ckpt
rm -f  /root/autodl-tmp/weights/GPT_weights_v2Pro/*.ckpt
rm -f  /root/autodl-tmp/weights/SoVITS_weights_v2Pro/*.pth
rm -rf $EXP_DIR/logs_s2_v2Pro/*.pth
mkdir -p $EXP_DIR/logs_s1_v2Pro

rm -f $LOG
nohup bash /root/autodl-tmp/_run_train.sh > $LOG 2>&1 &
echo "launched pid=$!  log=$LOG"
sleep 12
echo "=== first 25 lines ==="
head -25 $LOG 2>/dev/null
