# -*- coding: utf-8 -*-
"""导出 K=4 聚类中最接近目标音色的簇(イレイナ)到 radio_drama_final/，供 ASR 与训练。"""
import io
import json
import os
import shutil
import sys
import wave

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = r"radio_drama_work/radio_drama_fine"
DST = r"radio_drama_work/radio_drama_final"
RES = r"radio_drama_work/cluster_result.json"
K = "4"

d = json.load(open(RES, encoding="utf-8"))
names, assign, best = d["names"], d[K]["assign"], d[K]["best"]

if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST)

picked = [n for n, a in zip(names, assign) if a == best]
for n in picked:
    shutil.copy2(os.path.join(SRC, n), os.path.join(DST, n))

durs = []
for f in os.listdir(DST):
    with wave.open(os.path.join(DST, f)) as w:
        durs.append(w.getnframes() / w.getframerate())
durs.sort()
print(f"[export] 簇{best} -> {len(picked)} 段 -> {DST}")
print(
    "  总时长 %.1f 分钟 | min %.1fs 中位 %.1fs max %.1fs | <1.5s 的段: %d"
    % (sum(durs) / 60, durs[0], durs[len(durs) // 2], durs[-1], sum(1 for x in durs if x < 1.5))
)
