#!/bin/bash
set -u
PY=/root/miniconda3/envs/gptsovits/bin/python
echo "PY=$PY"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); import faster_whisper; print("faster_whisper ok")'
find /root/autodl-tmp/pkg -maxdepth 6 -type d -iname '*faster*whisper*' 2>/dev/null | head -30
du -sh /root/autodl-tmp/weights/* 2>/dev/null | tail -30
