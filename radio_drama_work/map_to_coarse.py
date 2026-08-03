# -*- coding: utf-8 -*-
"""把细切片的说话人聚类结果，按时间轴映射回完整的粗切句子。
目的：既保留完整句子(利于韵律建模)，又保证段内只有目标说话人。

用户听音判定：簇0(0.958) 与 簇3(0.746) 均为 イレイナ。
规则：对每个粗切段，统计落在其时间区间内的细切段，按时长加权计算"目标簇占比"，
占比 >= PURITY 的粗切段才保留。
"""
import io
import json
import os
import re
import shutil
import sys
import wave

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

COARSE = r"radio_drama_work/radio_drama"
FINE = r"radio_drama_work/radio_drama_fine"
DST = r"radio_drama_work/radio_drama_pure"
RES = r"radio_drama_work/cluster_result.json"
K = "4"
TARGETS = {0, 3}  # 用户听音判定的目标说话人簇
PURITY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85

d = json.load(open(RES, encoding="utf-8"))
names, assign = d["names"], d[K]["assign"]
fine_label = dict(zip(names, assign))

PAT = re.compile(r"^(.*)_(/d{10})_(/d{10})/.wav$")


def parse(n):
    m = PAT.match(n)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


fine = []
for n, a in fine_label.items():
    p = parse(n)
    if p:
        fine.append((p[0], p[1], p[2], a))

# 先做一次纯度统计（不落盘），扫描不同阈值
ratios = []
for cn in sorted(os.listdir(COARSE)):
    p = parse(cn)
    if not p:
        continue
    src, cs, ce = p
    tgt_len = other_len = 0
    for fs_src, fs, fe, a in fine:
        if fs_src != src:
            continue
        ov = min(ce, fe) - max(cs, fs)
        if ov <= 0:
            continue
        if a in TARGETS:
            tgt_len += ov
        else:
            other_len += ov
    total = tgt_len + other_len
    ratios.append((cn, tgt_len / total if total else 0.0, total))

print("=== 不同纯度阈值下的保留量 ===")
for th in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6):
    sel = [x for x in ratios if x[2] > 0 and x[1] >= th]
    dur = 0.0
    for cn, _, _ in sel:
        with wave.open(os.path.join(COARSE, cn)) as w:
            dur += w.getnframes() / w.getframerate()
    print("  purity>=%.2f -> %3d 段 / %.1f 分钟" % (th, len(sel), dur / 60))

if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST)

kept = [x for x in ratios if x[2] > 0 and x[1] >= PURITY]
for cn, _, _ in kept:
    shutil.copy2(os.path.join(COARSE, cn), os.path.join(DST, cn))

durs = []
for f in os.listdir(DST):
    with wave.open(os.path.join(DST, f)) as w:
        durs.append(w.getnframes() / w.getframerate())
durs.sort()

print()
print("[采用 purity>=%.2f] 粗切 %d 段 -> 保留 %d 段" % (PURITY, len(ratios), len(kept)))
if durs:
    print(
        "  保留总时长 %.1f 分钟 | min %.1fs 中位 %.1fs max %.1fs"
        % (sum(durs) / 60, durs[0], durs[len(durs) // 2], durs[-1])
    )
print("[out] %s" % DST)
