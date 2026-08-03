import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
CAND_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "new_candidates_sp75"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "new_multiref_scores.json"

os.chdir(PKG)
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
model.load_state_dict(torch.load(PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt", map_location="cpu"))
model.eval().to(device)
resampler = torchaudio.transforms.Resample(32000, 16000).to(device)


@torch.no_grad()
def embed(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    wav16 = resampler(wav.to(device))
    feats = torch.stack([Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav16])
    return torch.nn.functional.normalize(model.forward(feats), dim=-1).squeeze(0).cpu()


refs = torch.stack([embed(p) for p in sorted(REF_DIR.glob("*.wav"))])
center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
ref_center = (refs @ center).numpy()
negative_path = CAND_DIR / "28055832533-1-192_vocals_0017516160_0017573120.wav"
negative = embed(negative_path) if negative_path.exists() else None

rows = []
for path in sorted(CAND_DIR.glob("*.wav")):
    e = embed(path)
    sims = (refs @ e).numpy()
    row = {
        "name": path.name,
        "center": float(e @ center),
        "p10": float(np.quantile(sims, .10)),
        "p25": float(np.quantile(sims, .25)),
        "median": float(np.median(sims)),
        "p75": float(np.quantile(sims, .75)),
        "max": float(sims.max()),
        "spread": float(sims.std()),
    }
    if negative is not None:
        row["known_negative_sim"] = float(e @ negative)
        row["target_negative_margin"] = row["center"] - row["known_negative_sim"]
    rows.append(row)
rows.sort(key=lambda r: (r["p25"], r["median"], r["center"]), reverse=True)
OUT.write_text(json.dumps({"device": device, "ref_count": len(refs), "ref_center_min": float(ref_center.min()), "ref_center_median": float(np.median(ref_center)), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"device={device} refs={len(refs)} candidates={len(rows)}")
print(f"ref_center min={ref_center.min():.3f} median={np.median(ref_center):.3f} p10={np.quantile(ref_center,.10):.3f}")
for r in rows[:25]:
    print(f"{r['p25']:.3f}/t{r['median']:.3f}/t{r['center']:.3f}/t{r['max']:.3f}/t{r['name']}")
print(f"saved={OUT}")
