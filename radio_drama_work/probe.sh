#!/bin/bash
echo "===== SYSTEM ====="
uname -a
cat /etc/issue 2>/dev/null | head -2
echo
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>/dev/null || echo "no nvidia-smi"
echo
echo "===== CPU / RAM ====="
nproc
free -g | head -2
echo
echo "===== DISK ====="
df -h / /root /root/autodl-tmp 2>/dev/null | grep -v ^Filesystem | sort -u
echo
echo "===== HOME LAYOUT ====="
ls -la /root/ 2>/dev/null | head -30
echo
echo "--- autodl-tmp ---"
ls -la /root/autodl-tmp/ 2>/dev/null | head -30
echo
echo "===== LOOK FOR GPT-SoVITS ====="
for d in /root /root/autodl-tmp /root/autodl-fs /root/autodl-pub /workspace; do
  [ -d "$d" ] && find "$d" -maxdepth 3 -iname "*GPT*SoVITS*" -o -maxdepth 3 -iname "*gpt_sovits*" 2>/dev/null | head -10
done | sort -u | head -20
echo
echo "===== CONDA ENVS ====="
which conda && conda env list 2>/dev/null
echo
echo "===== PYTHON / TORCH ====="
python -V 2>&1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tail -2
echo
echo "===== NETWORK TURBO ====="
ls -la /etc/network_turbo 2>/dev/null && echo "network_turbo available" || echo "no network_turbo"
