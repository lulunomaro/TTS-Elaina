# -*- coding: utf-8 -*-
"""Remove user-confirmed mixed intervals from the experimental hard-cut output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio

ROOT = Path(r"radio_drama_work/radio_drama_work")
REPORT = ROOT / "word_target_hardcut_cluster_v1" / "word_target_hardcut_cluster_report.json"
OUT = ROOT / "word_target_hardcut_safe_eval_v1"
OUT.mkdir(parents=True, exist_ok=True)

# These are validation labels supplied by the user, not a replacement for the
# automatic detector.  Any word overlapping a known mixed interval is dropped.
MIXED = {
    "1405440540-1-192_vocals_0012969600_0013081600.wav": [(1.50, 3.00)],
    "28055832533-1-192_vocals_0013268480_0013539200.wav": [(5.50, 7.00)],
    "28055832533-1-192_vocals_0018346880_0018502400.wav": [(2.00, 3.50)],
}


def overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start < b and end > a for a, b in intervals)


def main() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    safe = {"note": "Experimental hard cuts after excluding user-confirmed mixed intervals.", "files": []}
    for row in payload["files"]:
        source = Path(row["source"])
        wav, sr = torchaudio.load(str(source))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        keep = [w for w in row["accepted_words"] if not overlaps(w["start"], w["end"], MIXED.get(source.name, []))]
        mask = np.zeros(wav.shape[1], dtype=np.float32)
        for w in keep:
            st = max(0, int(round((w["start"] - .055) * sr)))
            en = min(wav.shape[1], int(round((w["end"] + .055) * sr)))
            mask[st:en] = 1.0
        out_path = OUT / f"{source.stem}_safe_hardcut.wav"
        sf.write(str(out_path), (wav.numpy() * mask[None, :]).T, sr, subtype="PCM_16")
        words_dir = OUT / "accepted_words"
        words_dir.mkdir(exist_ok=True)
        word_files = []
        for idx, w in enumerate(keep):
            st = max(0, int(round((w["start"] - .055) * sr)))
            en = min(wav.shape[1], int(round((w["end"] + .055) * sr)))
            word_path = words_dir / f"{source.stem}_safe_word_{idx:03d}.wav"
            sf.write(str(word_path), wav[:, st:en].numpy().T, sr, subtype="PCM_16")
            word_files.append({"file": str(word_path), "start": w["start"], "end": w["end"], "text": w["text"]})
        safe["files"].append({"source": str(source), "hardcut_file": str(out_path), "accepted_after_mixed_exclusion": len(keep), "words": word_files})
        print(f"{source.name}: auto={len(row['accepted_words'])} safe={len(keep)}")
    (OUT / "safe_eval_report.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={OUT / 'safe_eval_report.json'}")


if __name__ == "__main__":
    main()
