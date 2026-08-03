# -*- coding: utf-8 -*-
"""切片广播剧纯净人声（内联 slice 逻辑，避免 slice_audio.py 的 CLI 副作用）。"""
import os
import sys
import traceback

import numpy as np
from scipy.io import wavfile

PKG = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "tools"))

from tools.my_utils import load_audio  # noqa: E402
from slicer2 import Slicer  # noqa: E402

INPUTS = [
    r"radio_drama_work/vocals/280479210-1-192_vocals.wav",
    r"radio_drama_work/vocals/113814583-1-160_vocals.wav",
]
OPT = r"radio_drama_work/sliced"

THRESHOLD = -40
MIN_LENGTH = 3000
MIN_INTERVAL = 300
HOP_SIZE = 20
MAX_SIL_KEPT = 500
_MAX = 0.9
ALPHA = 0.25

os.makedirs(OPT, exist_ok=True)
slicer = Slicer(
    sr=32000,
    threshold=THRESHOLD,
    min_length=MIN_LENGTH,
    min_interval=MIN_INTERVAL,
    hop_size=HOP_SIZE,
    max_sil_kept=MAX_SIL_KEPT,
)

total = 0
for inp_path in INPUTS:
    if not os.path.isfile(inp_path):
        print("[skip] not found:", inp_path)
        continue
    try:
        name = os.path.basename(inp_path)
        audio = load_audio(inp_path, 32000)
        print(f"[load] {name} samples={audio.shape} dur={len(audio)/32000:.1f}s")
        n = 0
        for chunk, start, end in slicer.slice(audio):
            tmp_max = np.abs(chunk).max()
            if tmp_max <= 0:
                continue
            if tmp_max > 1:
                chunk /= tmp_max
            chunk = (chunk / tmp_max * (_MAX * ALPHA)) + (1 - ALPHA) * chunk
            wavfile.write(
                "%s/%s_%010d_%010d.wav" % (OPT, name, start, end),
                32000,
                (chunk * 32767).astype(np.int16),
            )
            n += 1
        print(f"[slice] {name} -> {n} chunks")
        total += n
    except Exception:
        print(inp_path, "->fail->", traceback.format_exc())

print(f"[done] total chunks = {total}")
