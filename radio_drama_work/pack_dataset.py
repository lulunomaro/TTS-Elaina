# -*- coding: utf-8 -*-
"""把合并数据集打平成 dataset/wavs/ + list（服务器路径），并压成 zip 以便上传。"""
import os
import shutil
import zipfile
import wave

MERGED = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50/logs/my_voice_merged/merged.list"
OUT = r"radio_drama_work/dataset_v1"
WAVS = os.path.join(OUT, "wavs")
REMOTE_ROOT = "/root/autodl-tmp/dataset_v1"

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(WAVS)

rows = []
seen = set()
n_old = n_radio = 0
dur = 0.0

for ln in open(MERGED, encoding="utf-8"):
    p = ln.rstrip("/n").split("|")
    if len(p) < 4:
        continue
    src, spk, lang, text = p[0], p[1], p[2], p[3]
    if not os.path.exists(src):
        continue
    is_radio = "radio_drama_pure" in src.lower()
    prefix = "radio" if is_radio else "old"
    base = "%s_%s" % (prefix, os.path.basename(src))
    # 防重名
    i = 1
    while base in seen:
        base = "%s_%d_%s" % (prefix, i, os.path.basename(src))
        i += 1
    seen.add(base)
    shutil.copy2(src, os.path.join(WAVS, base))
    try:
        w = wave.open(src)
        dur += w.getnframes() / w.getframerate()
        w.close()
    except Exception:
        pass
    if is_radio:
        n_radio += 1
    else:
        n_old += 1
    rows.append("%s/wavs/%s|%s|%s|%s" % (REMOTE_ROOT, base, spk, lang, text))

list_path = os.path.join(OUT, "dataset.list")
with open(list_path, "w", encoding="utf-8") as f:
    f.write("/n".join(rows) + "/n")

print("segments: %d (old %d + radio %d), dur %.1f min" % (len(rows), n_old, n_radio, dur / 60))

zip_path = r"radio_drama_work/dataset_v1.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
    z.write(list_path, "dataset_v1/dataset.list")
    for fn in sorted(os.listdir(WAVS)):
        z.write(os.path.join(WAVS, fn), "dataset_v1/wavs/" + fn)
print("zip: %s  %.1f MB" % (zip_path, os.path.getsize(zip_path) / 1048576))
