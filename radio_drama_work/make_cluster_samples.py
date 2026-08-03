# -*- coding: utf-8 -*-
"""为每个聚类簇生成一段试听拼接音频(段间插 0.4s 静音)，供人工判断哪些簇是目标音色。
同时输出每簇的 ASR 文本样例。"""
import io
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FINE = r"radio_drama_work/radio_drama_fine"
RES = r"radio_drama_work/cluster_result.json"
OUT = r"radio_drama_work/cluster_samples"
REF_SRC = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50/logs/my_voice/5-wav32k"
K = "4"
SR = 32000
MAX_SEC = 35

os.makedirs(OUT, exist_ok=True)
d = json.load(open(RES, encoding="utf-8"))
names, assign, csim = d["names"], d[K]["assign"], d[K]["cluster_sim"]


def concat(paths, out_path, max_sec=MAX_SEC):
    gap = np.zeros(int(SR * 0.4), dtype=np.int16)
    buf, tot = [], 0
    for p in paths:
        sr, x = wavfile.read(p)
        if x.ndim > 1:
            x = x.mean(1).astype(np.int16)
        buf.append(x)
        buf.append(gap)
        tot += len(x) / sr + 0.4
        if tot >= max_sec:
            break
    if buf:
        wavfile.write(out_path, SR, np.concatenate(buf))
    return tot


# 参考样本（老数据 = 目标音色）
ref_files = sorted(os.listdir(REF_SRC))[:20]
t = concat([os.path.join(REF_SRC, f) for f in ref_files], os.path.join(OUT, "REF_目标音色_老数据.wav"))
print(f"[ref] REF_目标音色_老数据.wav  {t:.0f}s")

for j in sorted(set(assign), key=lambda x: -csim[x]):
    sel = [n for n, a in zip(names, assign) if a == j]
    # 取簇内均匀采样，避免只取同一段落
    step = max(1, len(sel) // 25)
    picks = sel[::step]
    fn = f"簇{j}_{len(sel)}段_相似度{csim[j]:.3f}.wav"
    t = concat([os.path.join(FINE, p) for p in picks], os.path.join(OUT, fn))
    print(f"[cluster] {fn}  {t:.0f}s")

print(f"/n[out] {OUT}")
