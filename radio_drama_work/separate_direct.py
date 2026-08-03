#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows 下直接用 BS-Roformer 分离，绕过 handoff uvr5_cli.py 的 /tmp 重采样坑。
前提：输入 wav 已是 44100Hz 立体声（已用 ffmpeg 抽好）。"""
import os, sys
TMP = "F:/GPTSOVITS/radio_drama_work/tmp"
os.environ["TEMP"] = TMP
os.environ["TMP"] = TMP
os.makedirs(TMP, exist_ok=True)

PKG = "F:/GPTSOVITS/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
os.chdir(PKG)
sys.path.insert(0, os.path.join(PKG, "tools/uvr5"))

from bsroformer import Roformer_Loader

device = "cuda:0"
is_half = False
WEIGHT_ROOT = os.path.join(PKG, "tools/uvr5/uvr5_weights")
MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755"

print("加载 BS-Roformer (fp32): %s" % MODEL_NAME, flush=True)
pre_fun = Roformer_Loader(
    model_path=os.path.join(WEIGHT_ROOT, MODEL_NAME + ".ckpt"),
    config_path=os.path.join(WEIGHT_ROOT, MODEL_NAME + ".yaml"),  # 不存在 -> 默认配置
    device=device,
    is_half=is_half,
)

raw = "F:/GPTSOVITS/radio_drama_work/raw"
vocal = "F:/GPTSOVITS/radio_drama_work/vocals"
ins = "F:/GPTSOVITS/radio_drama_work/ins"
os.makedirs(vocal, exist_ok=True)
os.makedirs(ins, exist_ok=True)

files = sorted([f for f in os.listdir(raw) if f.lower().endswith((".wav", ".mp3", ".flac"))])
print("待分离: %d 个文件" % len(files), flush=True)
for i, name in enumerate(files):
    inp = os.path.join(raw, name)
    pre_fun._path_audio_(inp, ins, vocal, "wav", False)
    print("[%d/%d] done: %s" % (i + 1, len(files), name), flush=True)
print("BS-Roformer 人声分离完成 -> %s" % vocal, flush=True)
