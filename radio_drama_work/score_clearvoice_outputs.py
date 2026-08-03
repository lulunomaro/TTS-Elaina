# -*- coding: utf-8 -*-
"""Score ClearVoice separation stems against known Ireina references."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torchaudio

ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
OUT_DIRS = [
    ROOT / "clearvoice_test" / "outputs",
    ROOT / "clearvoice_test" / "outputs_v2",
    ROOT / "clearvoice_test" / "outputs_windows",
]
if os.environ.get("CLEARVOICE_SCORE_DIR"):
    OUT_DIRS = [Path(os.environ["CLEARVOICE_SCORE_DIR"])]
OUT_JSON = Path(os.environ.get("CLEARVOICE_SCORE_JSON", str(ROOT / "clearvoice_test" / "clearvoice_scores.json")))
REF_DIR = PKG / "logs" / "my_voice" / "5-wav32k"
NEGATIVE_LIST = ROOT / "confirmed_non_target_or_mixed.txt"

os.chdir(PKG)
sys.path.insert(0, str(PKG / "GPT_SoVITS" / "eres2net"))
from ERes2NetV2 import ERes2NetV2  # noqa: E402
import kaldi as Kaldi  # noqa: E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
model.load_state_dict(torch.load(PKG / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt", map_location="cpu"))
model.eval().to(device)


@torch.no_grad()
def embed(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.to(device)
    feat = torch.stack([Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav])
    return torch.nn.functional.normalize(model.forward(feat), dim=-1).squeeze(0).cpu()


def negative_paths() -> list[Path]:
    paths = []
    for line in NEGATIVE_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = Path(line.split("|", 1)[0].strip())
            if path.exists():
                paths.append(path)
    return paths


def main() -> None:
    refs = torch.stack([embed(p) for p in sorted(REF_DIR.glob("*.wav"))])
    center = torch.nn.functional.normalize(refs.mean(0), dim=-1)
    negs = torch.stack([embed(p) for p in negative_paths()])
    rows = []
    paths = sorted({path for out_dir in OUT_DIRS for path in out_dir.rglob("*.wav")})
    for path in paths:
        emb = embed(path)
        vals = emb @ refs.T
        neg_vals = emb @ negs.T
        row = {
            "file": str(path),
            "center": float(emb @ center),
            "median": float(vals.median()),
            "p25": float(torch.quantile(vals, .25)),
            "p75": float(torch.quantile(vals, .75)),
            "known_negative_sim": float(neg_vals.max()),
        }
        row["target_negative_margin"] = row["center"] - row["known_negative_sim"]
        rows.append(row)
    payload = {"device": device, "reference_count": len(refs), "negative_count": len(negs), "files": rows}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['center']:.4f}/t{row['median']:.4f}/t{row['p25']:.4f}/t{row['known_negative_sim']:.4f}/t{row['target_negative_margin']:.4f}/t{row['file']}")
    print(f"saved={OUT_JSON}")


if __name__ == "__main__":
    main()
