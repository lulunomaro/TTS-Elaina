# -*- coding: utf-8 -*-
"""说话人聚类：对细切片提 ERes2NetV2 声纹 embedding，做球面 K-means 聚类，
看能否把广播剧里的不同角色分开，并挑出与老数据(目标音色)最接近的簇。
输出 embedding 缓存 + 聚类结果，供后续筛选。"""
import io
import json
import os
import sys

import torch
import torchaudio

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PKG = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
os.chdir(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "GPT_SoVITS", "eres2net"))

from ERes2NetV2 import ERes2NetV2  # noqa: E402
import kaldi as Kaldi  # noqa: E402

SV_PATH = os.path.join(PKG, "GPT_SoVITS", "pretrained_models", "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
REF_DIR = os.path.join(PKG, "logs", "my_voice", "5-wav32k")
CAND_DIR = r"radio_drama_work/radio_drama_fine"
EMB_CACHE = r"radio_drama_work/emb_fine.pt"
OUT_JSON = r"radio_drama_work/cluster_result.json"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
model.load_state_dict(torch.load(SV_PATH, map_location="cpu"))
model.eval().to(device)
resampler = torchaudio.transforms.Resample(32000, 16000).to(device)


@torch.no_grad()
def embed(p):
    wav, sr = torchaudio.load(p)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(device)
    w16 = resampler(wav)
    feat = torch.stack([Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in w16])
    return torch.nn.functional.normalize(model.forward(feat), dim=-1).squeeze(0).cpu()


if os.path.exists(EMB_CACHE):
    d = torch.load(EMB_CACHE)
    names, X, center = d["names"], d["X"], d["center"]
    print(f"[cache] 载入 {len(names)} 条 embedding")
else:
    ref = [embed(os.path.join(REF_DIR, f)) for f in sorted(os.listdir(REF_DIR)) if f.endswith(".wav")]
    center = torch.nn.functional.normalize(torch.stack(ref).mean(0), dim=-1)
    names = sorted(f for f in os.listdir(CAND_DIR) if f.endswith(".wav"))
    X = torch.stack([embed(os.path.join(CAND_DIR, f)) for f in names])
    torch.save({"names": names, "X": X, "center": center}, EMB_CACHE)
    print(f"[embed] {len(names)} 条完成，已缓存")


def spherical_kmeans(X, k, iters=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    # k-means++ 风格初始化
    idx = [torch.randint(len(X), (1,), generator=g).item()]
    for _ in range(k - 1):
        d = 1 - (X @ X[idx].T).max(dim=1).values
        idx.append(int(torch.multinomial(d.clamp(min=1e-8), 1, generator=g)))
    C = X[idx].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(dim=1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = torch.nn.functional.normalize(X[m].mean(0), dim=-1)
    return a, C


print(f"/n参考中心与候选整体相似度: 中位 {(X @ center).median():.3f}/n")

results = {}
for k in [2, 3, 4]:
    a, C = spherical_kmeans(X, k)
    csim = (C @ center).tolist()  # 每个簇心与目标音色的相似度
    print(f"===== K={k} =====")
    order = sorted(range(k), key=lambda j: -csim[j])
    for rank, j in enumerate(order):
        m = a == j
        cnt = int(m.sum())
        inner = float((X[m] @ C[j]).mean()) if cnt else 0
        tag = "  <== 最接近目标音色" if rank == 0 else ""
        print(f"  簇{j}: {cnt:3d} 段 | 簇心↔目标 {csim[j]:.3f} | 簇内紧致度 {inner:.3f}{tag}")
    results[str(k)] = {"assign": a.tolist(), "cluster_sim": csim, "best": order[0]}
    print()

json.dump({"names": names, **results}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False)
print(f"[saved] {OUT_JSON}")
