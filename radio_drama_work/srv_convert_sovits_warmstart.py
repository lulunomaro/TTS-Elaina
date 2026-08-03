#!/usr/bin/env python3
"""Convert a v2Pro exported SoVITS weight into trainer warm-start format.

v2Pro exported weights carry a two-byte model-version header.  The official
loader restores the missing ZIP header before torch.load; the trainer expects
only a dict containing the generator state dict under ``weight``.
"""
import os
import sys

import torch

REPO = "/root/GPT-SoVITS"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
from GPT_SoVITS.process_ckpt import load_sovits_new

src = sys.argv[1]
dst = sys.argv[2]
obj = load_sovits_new(src)
if not isinstance(obj, dict) or "weight" not in obj:
    raise RuntimeError(f"unexpected checkpoint object: {type(obj)}")
state = obj["weight"]
if not isinstance(state, dict) or not state:
    raise RuntimeError("checkpoint has no generator state dict")
os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save({"weight": state, "info": obj.get("info", "warm-start")}, dst)
print(f"converted={dst} tensors={len(state)} size={os.path.getsize(dst)}")
