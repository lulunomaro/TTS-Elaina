# -*- coding: utf-8 -*-
"""构建"老 164 段 + 广播剧纯净 110 段"的合并训练清单。

老数据: logs/my_voice/asr_opt/vocals.list (相对路径 -> 转绝对)
新数据: radio_drama_pure/ 下的粗切段，文本取自 asr_out/radio_drama.list
输出:   logs/my_voice_merged/merged.list  (绝对路径|说话人|JA|文本)
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
OLD_LIST = os.path.join(ROOT, "logs", "my_voice", "asr_opt", "vocals.list")
PURE_DIR = r"radio_drama_work/radio_drama_pure"
NEW_LIST = r"radio_drama_work/asr_out/radio_drama.list"
EXP_DIR = os.path.join(ROOT, "logs", "my_voice_merged")
OUT = os.path.join(EXP_DIR, "merged.list")

os.makedirs(EXP_DIR, exist_ok=True)

rows = []
miss_old = 0
for ln in open(OLD_LIST, encoding="utf-8"):
    p = ln.rstrip("/n").split("|")
    if len(p) < 4 or not p[3].strip():
        continue
    ap = os.path.normpath(os.path.join(ROOT, p[0]))
    if not os.path.exists(ap):
        miss_old += 1
        continue
    rows.append((ap, "old", "JA", p[3]))
n_old = len(rows)

keep = set(os.listdir(PURE_DIR))
text = {}
for ln in open(NEW_LIST, encoding="utf-8"):
    p = ln.rstrip("/n").split("|")
    if len(p) >= 4:
        text[os.path.basename(p[0].replace("//", "/"))] = p[3]

miss_new = 0
for b in sorted(keep):
    t = text.get(b, "").strip()
    if not t:
        miss_new += 1
        continue
    rows.append((os.path.join(PURE_DIR, b), "rd", "JA", t))
n_new = len(rows) - n_old

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write("|".join(r) + "/n")

print("老数据 %d 段 (缺失 %d) + 广播剧 %d 段 (无文本 %d) = 合计 %d 段" % (n_old, miss_old, n_new, miss_new, len(rows)))
print("[out]", OUT)
print("[exp_dir]", EXP_DIR)
