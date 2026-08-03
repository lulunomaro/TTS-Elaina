# -*- coding: utf-8 -*-
"""使用现有 BS-Roformer 分离新增广播剧音轨；输入视频已先抽成 WAV。"""

import os
import sys


TMP = "F:/GPTSOVITS/radio_drama_work/tmp_new_broadcast"
PKG = "F:/GPTSOVITS/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
FFMPEG_DIR = os.path.join(PKG, "runtime")
os.environ["TEMP"] = TMP
os.environ["TMP"] = TMP
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
os.makedirs(TMP, exist_ok=True)

os.chdir(PKG)
sys.path.insert(0, os.path.join(PKG, "tools/uvr5"))
from bsroformer import Roformer_Loader  # noqa: E402


MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755"
WEIGHT_ROOT = os.path.join(PKG, "tools/uvr5/uvr5_weights")
raw = "F:/GPTSOVITS/radio_drama_work/new_raw"
vocal = "F:/GPTSOVITS/radio_drama_work/new_vocals"
ins = "F:/GPTSOVITS/radio_drama_work/new_ins"
os.makedirs(vocal, exist_ok=True)
os.makedirs(ins, exist_ok=True)

print(f"loading {MODEL_NAME}", flush=True)
pre_fun = Roformer_Loader(
    model_path=os.path.join(WEIGHT_ROOT, MODEL_NAME + ".ckpt"),
    config_path=os.path.join(WEIGHT_ROOT, MODEL_NAME + ".yaml"),
    device="cuda:0",
    is_half=False,
)
files = sorted(f for f in os.listdir(raw) if f.lower().endswith(".wav"))
print(f"inputs={len(files)}", flush=True)
for i, name in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {name}", flush=True)
    pre_fun._path_audio_(os.path.join(raw, name), ins, vocal, "wav", False)
print(f"done vocals={vocal}", flush=True)
