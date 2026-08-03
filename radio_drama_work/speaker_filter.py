# -*- coding: utf-8 -*-
"""说话人筛选：用 ERes2NetV2 声纹模型，把广播剧切片与老 164 段(目标音色 イレイナ)比对，
输出每段的余弦相似度，供设定阈值剔除其他角色。只做分析+报告，不删文件。"""
import os
import sys
import json

import torch
import torchaudio

PKG = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
os.chdir(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "GPT_SoVITS", "eres2net"))

from ERes2NetV2 import ERes2NetV2  # noqa: E402
import kaldi as Kaldi  # noqa: E402

SV_PATH = os.path.join(PKG, "GPT_SoVITS", "pretrained_models", "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
REF_DIR = os.path.join(PKG, "logs", "my_voice", "5-wav32k")          # 老 164 段(目标音色)
CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else r"radio_drama_work/radio_drama"
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else r"radio_drama_work/speaker_scores.json"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[init] device={device}")

model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
model.load_state_dict(torch.load(SV_PATH, map_location="cpu"))
model.eval().to(device)
resampler = torchaudio.transforms.Resample(32000, 16000).to(device)


@torch.no_grad()
def embed(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 32000:
        wav = torchaudio.transforms.Resample(sr, 32000)(wav)
    wav = wav.to(device)
    wav16 = resampler(wav)
    feat = torch.stack(
        [Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav16]
    )
    emb = model.forward(feat)  # (1, D) 标准说话人 embedding
    return torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu()


# 1) 目标音色中心
ref_files = sorted(f for f in os.listdir(REF_DIR) if f.lower().endswith(".wav"))
print(f"[ref] {len(ref_files)} 段参考音频")
ref_embs = []
for i, f in enumerate(ref_files):
    try:
        ref_embs.append(embed(os.path.join(REF_DIR, f)))
    except Exception as e:
        print("  ref fail:", f, e)
ref_mat = torch.stack(ref_embs)
center = torch.nn.functional.normalize(ref_mat.mean(0), dim=-1)

ref_sims = (ref_mat @ center).tolist()
ref_sims_sorted = sorted(ref_sims)
n = len(ref_sims_sorted)
print("/n[基线] 老数据自身与中心的相似度分布（同一人应该都很高）：")
print(
    "  min %.3f  p05 %.3f  p25 %.3f  中位 %.3f  max %.3f"
    % (
        ref_sims_sorted[0],
        ref_sims_sorted[int(n * 0.05)],
        ref_sims_sorted[int(n * 0.25)],
        ref_sims_sorted[n // 2],
        ref_sims_sorted[-1],
    )
)

# 2) 候选打分
cand_files = sorted(f for f in os.listdir(CAND_DIR) if f.lower().endswith(".wav"))
print(f"/n[cand] {len(cand_files)} 段广播剧切片")
scores = {}
for f in cand_files:
    try:
        e = embed(os.path.join(CAND_DIR, f))
        scores[f] = float(e @ center)
    except Exception as ex:
        print("  cand fail:", f, ex)

vals = sorted(scores.values())
m = len(vals)
print("/n[候选] 与目标音色中心的相似度分布：")
print(
    "  min %.3f  p10 %.3f  p25 %.3f  中位 %.3f  p75 %.3f  p90 %.3f  max %.3f"
    % (
        vals[0],
        vals[int(m * 0.10)],
        vals[int(m * 0.25)],
        vals[m // 2],
        vals[int(m * 0.75)],
        vals[int(m * 0.90)],
        vals[-1],
    )
)

print("/n[直方图] 相似度区间 -> 段数")
edges = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
for a, b in zip(edges[:-1], edges[1:]):
    c = sum(1 for v in vals if a <= v < b)
    if c:
        print(f"  [{a:>5.2f}, {b:>4.2f}) {'#' * max(1, c // 3):<40} {c}")

print("/n[阈值预览] 保留段数：")
for th in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    keep = sum(1 for v in vals if v >= th)
    print(f"  >= {th:.2f} -> 保留 {keep:3d} / {m}  ({keep / m * 100:.0f}%)")

with open(OUT_JSON, "w", encoding="utf-8") as fp:
    json.dump({"center_ref_sims": ref_sims, "cand_scores": scores}, fp, ensure_ascii=False, indent=1)
print(f"/n[saved] {OUT_JSON}")
