# -*- coding: utf-8 -*-
"""交叉验证：把声纹相似度和 ASR 文本对照，人工判断低分段是否为其他角色。"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCORES = r"radio_drama_work/speaker_scores.json"
LIST = r"radio_drama_work/asr_out/radio_drama.list"

S = json.load(open(SCORES, encoding="utf-8"))["cand_scores"]

T = {}
for ln in open(LIST, encoding="utf-8"):
    p = ln.rstrip("/n").split("|")
    if len(p) >= 4:
        T[os.path.basename(p[0].replace("//", "/"))] = p[3]

srt = sorted(S.items(), key=lambda x: x[1])


def show(title, items):
    print(f"===== {title} =====")
    for k, v in items:
        print("  %.3f | %s" % (v, T.get(k, "(无文本)")[:50]))
    print()


show("最低分 12 段", srt[:12])
show("最高分 10 段", srt[-10:])
show("0.58~0.63 边界 8 段", [x for x in srt if 0.58 <= x[1] <= 0.63][:8])

# Whisper 幻觉/无效文本检测
HALLU = ["ご視聴ありがとうございました", "ありがとうございました", "チャンネル登録", "字幕", "おわり", "END"]
bad = []
for k, v in S.items():
    t = T.get(k, "").strip()
    if not t:
        bad.append((k, v, "空文本"))
    elif len(t) <= 2:
        bad.append((k, v, f"过短:{t}"))
    elif any(h in t for h in HALLU):
        bad.append((k, v, f"疑似幻觉:{t[:30]}"))
print("===== 疑似无效/幻觉文本 =====")
for k, v, why in sorted(bad, key=lambda x: x[1]):
    print("  %.3f | %s" % (v, why))
print(f"  共 {len(bad)} 段")
