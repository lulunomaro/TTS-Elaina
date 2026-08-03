#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/gptsovits/bin/python
IN=/root/autodl-tmp/new_candidates_sp75
OUT=/root/autodl-tmp/new_candidates_fine_sv.json
PKG=/root/autodl-tmp/pkg/GPT-SoVITS-v2pro-20250604-nvidia50
"$PY" - "$IN" "$OUT" "$PKG" <<'PY'
import json, os, sys
import torch
import torchaudio

inp, out_path, pkg = sys.argv[1:]
os.chdir(pkg)
sys.path.insert(0, os.path.join(pkg, "GPT_SoVITS", "eres2net"))
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi

sv_path = os.path.join(pkg, "GPT_SoVITS", "pretrained_models", "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
ref_dir = os.path.join(pkg, "logs", "my_voice", "5-wav32k")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device", device, flush=True)
model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
model.load_state_dict(torch.load(sv_path, map_location="cpu"))
model.eval().to(device)
resampler = torchaudio.transforms.Resample(32000, 16000).to(device)

@torch.no_grad()
def embed_wave(wav):
    wav = wav.to(device)
    wav16 = resampler(wav)
    feat = torch.stack([Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav16])
    return torch.nn.functional.normalize(model.forward(feat), dim=-1)

refs = []
for name in sorted(os.listdir(ref_dir)):
    if name.lower().endswith(".wav"):
        wav, sr = torchaudio.load(os.path.join(ref_dir, name))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != 32000: wav = torchaudio.functional.resample(wav, sr, 32000)
        refs.append(embed_wave(wav))
center = torch.nn.functional.normalize(torch.cat(refs, 0).mean(0), dim=-1)

rows = []
for name in sorted(os.listdir(inp)):
    if not name.lower().endswith(".wav"): continue
    wav, sr = torchaudio.load(os.path.join(inp, name))
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if sr != 32000: wav = torchaudio.functional.resample(wav, sr, 32000)
    n = wav.shape[1]
    win, hop = 2 * 32000, 1 * 32000
    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] + win < n: starts.append(n - win)
    if n <= win: starts = [0]
    chunks, meta = [], []
    for st in starts:
        en = min(n, st + win)
        c = wav[:, st:en]
        if c.shape[1] < win: c = torch.nn.functional.pad(c, (0, win - c.shape[1]))
        chunks.append(c)
        meta.append((st / 32000.0, en / 32000.0))
    sims = []
    for i in range(0, len(chunks), 16):
        batch = torch.cat(chunks[i:i+16], 0)
        emb = embed_wave(batch)
        sims.extend((emb @ center).detach().cpu().tolist())
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    best = order[0]
    vals = sorted(sims)
    rows.append(dict(name=name, windows=[dict(start=meta[i][0], end=meta[i][1], score=float(sims[i])) for i in order],
                     best_score=float(sims[best]), best_start=meta[best][0], best_end=meta[best][1],
                     median=float(vals[len(vals)//2]), fraction_ge_075=sum(v >= .75 for v in sims)/len(sims)))
    print(f"{name}/tbest={sims[best]:.3f}/tmedian={vals[len(vals)//2]:.3f}/tfrac075={rows[-1]['fraction_ge_075']:.2f}", flush=True)

with open(out_path, "w", encoding="utf-8") as f: json.dump(rows, f, ensure_ascii=False, indent=2)
print("saved", out_path, len(rows), flush=True)
PY
