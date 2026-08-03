#!/bin/bash
echo "===== 实测 CUDA 计算能力 (sm_120) ====="
for E in GPTSoVits gptsovits; do
  echo "--- $E ---"
  /root/miniconda3/envs/$E/bin/python -c "
import torch,warnings
warnings.filterwarnings('ignore')
print('  arch_list:', torch.cuda.get_arch_list())
try:
    a=torch.randn(2048,2048,device='cuda'); b=torch.randn(2048,2048,device='cuda')
    c=(a@b).sum().item()
    print('  REAL MATMUL OK ->', round(c,2))
except Exception as e:
    print('  REAL MATMUL FAILED:', str(e)[:150])
" 2>&1 | grep -E "arch_list|REAL"
done
echo
echo "===== 磁盘可清理项 ====="
du -sh /root/autodl-tmp/gptsovits_handoff.zip /root/autodl-tmp/pkg /root/autodl-tmp/handoff_build /root/GPT-SoVITS/TEMP 2>/dev/null
echo "--- 系统盘占用 top ---"
du -sh /root/GPT-SoVITS /root/miniconda3 /root/.cache 2>/dev/null
echo
echo "===== 现有 logs / 已训权重 ====="
ls -la /root/GPT-SoVITS/logs/ 2>/dev/null | head -10
echo "--- v2Pro 权重目录 ---"
ls -la /root/GPT-SoVITS/SoVITS_weights_v2Pro/ /root/GPT-SoVITS/GPT_weights_v2Pro/ 2>/dev/null
echo
echo "===== 官方 webui 推荐轮数 ====="
grep -nE "default_batch_size|default_max_batch_size|value=8|value=15|总训练轮数|每张显卡的batch" /root/GPT-SoVITS/webui.py 2>/dev/null | head -12
echo
echo "===== 关键脚本存在性 ====="
ls /root/GPT-SoVITS/GPT_SoVITS/prepare_datasets/ 2>/dev/null
ls /root/GPT-SoVITS/GPT_SoVITS/s2_train.py /root/GPT-SoVITS/GPT_SoVITS/s1_train.py 2>/dev/null
echo
echo "===== ffmpeg / uvr ====="
which ffmpeg ffprobe 2>/dev/null
ls -d /root/GPT-SoVITS/tools/uvr5 2>/dev/null && ls /root/GPT-SoVITS/tools/uvr5/uvr5_weights/ 2>/dev/null | head
