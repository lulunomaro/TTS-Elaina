#!/bin/bash
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>/dev/null || echo "no nvidia-smi"
echo "===== CPU/RAM ====="
echo "cores: $(nproc)"; free -g | head -2
echo "===== DISK ====="
df -h /root /root/autodl-tmp 2>/dev/null
echo
echo "===== CONDA ENVS ====="
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda env list
echo
for E in GPTSoVits gptsovits; do
  echo "--- env: $E ---"
  P=/root/miniconda3/envs/$E/bin/python
  if [ -x "$P" ]; then
    $P -V 2>&1
    $P -c "import torch;print(' torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());print(' gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | tail -3
    $P -c "import importlib;[print('  %-14s %s'%(m,getattr(importlib.import_module(m),'__version__','?'))) for m in ['numpy','librosa','transformers','peft','gradio']]" 2>&1 | tail -6
  else
    echo " (no python)"
  fi
done
echo
echo "===== /root/GPT-SoVITS 仓库 ====="
cd /root/GPT-SoVITS && git log --oneline -3 2>/dev/null; git describe --tags 2>/dev/null
ls | head -30
echo "--- 预训练模型 ---"
du -sh GPT_SoVITS/pretrained_models 2>/dev/null
ls GPT_SoVITS/pretrained_models/ 2>/dev/null
echo "--- v2Pro 权重 ---"
ls -la GPT_SoVITS/pretrained_models/v2Pro/ 2>/dev/null
echo
echo "===== pkg 下的 v2pro 包 ====="
du -sh /root/autodl-tmp/pkg/GPT-SoVITS-v2pro-20250604-nvidia50 2>/dev/null
echo
echo "===== 已有实验数据 (logs) ====="
ls /root/GPT-SoVITS/logs/ 2>/dev/null | head
ls /root/GPT-SoVITS/SoVITS_weights* /root/GPT-SoVITS/GPT_weights* -d 2>/dev/null
echo
echo "===== 广播剧素材 ====="
ls -la /root/autodl-tmp/*.mp3 2>/dev/null
