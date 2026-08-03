# -*- coding: utf-8 -*-
"""细粒度切片：用更小的 min_interval / min_length，把对话中两个角色的换手处切开，
便于后续用声纹严格筛出目标说话人。"""
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
OPT = r"radio_drama_work/radio_drama_fine"

# 细切参数：更短最小段长 + 更敏感的静音间隔，用于切开对话换手
THRESHOLD = -40
MIN_LENGTH = 1800
MIN_INTERVAL = 120
HOP_SIZE = 10
MAX_SIL_KEPT = 250
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
    try:
        name = os.path.basename(inp_path)
        audio = load_audio(inp_path, 32000)
        n = 0
        for chunk, start, end in slicer.slice(audio):
            tmp_max = np.abs(chunk).max()
            if tmp_max <= 0:
                continue
            if tmp_max > 1:
                chunk /= tmp_max
            chunk = (chunk / tmp_max * (_MAX * ALPHA)) + (1 - ALPHA) * chunk
            wavfile.write("%s/%s_%010d_%010d.wav" % (OPT, name, start, end), 32000, (chunk * 32767).astype(np.int16))
            n += 1
        print(f"[slice] {name} -> {n} chunks")
        total += n
    except Exception:
        print(inp_path, "->fail->", traceback.format_exc())

print(f"[done] total = {total}")

# 时长分布
import wave  # noqa: E402

durs = []
for f in os.listdir(OPT):
    with wave.open(os.path.join(OPT, f)) as w:
        durs.append(w.getnframes() / w.getframerate())
durs.sort()
print(
    "时长: 总 %.1f 分钟 | min %.1fs p25 %.1fs 中位 %.1fs p75 %.1fs max %.1fs"
    % (sum(durs) / 60, durs[0], durs[len(durs) // 4], durs[len(durs) // 2], durs[len(durs) * 3 // 4], durs[-1])
)
