# -*- coding: utf-8 -*-
"""Extract and BS-Roformer-separate the newly supplied radio dramas.

This batch is intentionally isolated from the earlier ``new_raw`` and
``new_vocals`` directories.  The downstream speaker mining stage must only
read the generated vocals stems, never the mixed MP4 audio.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
BATCH = ROOT / "new_radio_batch_v1"
RAW = BATCH / "raw"
VOCALS = BATCH / "vocals"
OTHER = BATCH / "other"
TMP = BATCH / "tmp"
MANIFEST = BATCH / "manifest.json"

INPUTS = [
    Path(r"C:/Users/lulu/Downloads/37141415568-1-192.mp4"),
    Path(r"C:/Users/lulu/Downloads/28243657968-1-192.mp4"),
    Path(r"C:/Users/lulu/Downloads/37239655783-1-192.mp4"),
    Path(r"C:/Users/lulu/Downloads/1405372904-1-192.mp4"),
    Path(r"C:/Users/lulu/Downloads/306672160-1-192.mp4"),
    Path(r"C:/Users/lulu/Downloads/243358894-1-192.mp4"),
    Path(r"C:/Users/lulu/Downloads/253810599-1-192.mp4"),
]


def run_ffmpeg(source: Path, target: Path) -> None:
    ffmpeg = PKG / "runtime" / "ffmpeg.exe"
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn",
        "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    missing = [str(path) for path in INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing supplied videos: " + ", ".join(missing))

    for path in (RAW, VOCALS, OTHER, TMP):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(TMP)
    os.environ["TMP"] = str(TMP)
    os.environ["PATH"] = str(PKG / "runtime") + os.pathsep + os.environ.get("PATH", "")

    manifest = {"version": "new_radio_batch_v1", "inputs": [], "separation_model": "model_bs_roformer_ep_317_sdr_12.9755"}
    for index, source in enumerate(INPUTS, 1):
        stem = source.stem
        raw = RAW / f"{stem}.wav"
        vocal = VOCALS / f"{stem}_vocals.wav"
        other = OTHER / f"{stem}_other.wav"
        print(f"[{index}/{len(INPUTS)}] extract {source.name}", flush=True)
        if not raw.exists():
            run_ffmpeg(source, raw)
        manifest["inputs"].append({
            "source": str(source),
            "raw": str(raw),
            "vocals": str(vocal),
            "other": str(other),
        })

    os.chdir(PKG)
    sys.path.insert(0, str(PKG / "tools" / "uvr5"))
    from bsroformer import Roformer_Loader  # noqa: E402

    model_name = "model_bs_roformer_ep_317_sdr_12.9755"
    print(f"load BS-Roformer {model_name}", flush=True)
    loader = Roformer_Loader(
        model_path=str(PKG / "tools" / "uvr5" / "uvr5_weights" / f"{model_name}.ckpt"),
        config_path=str(PKG / "tools" / "uvr5" / "uvr5_weights" / f"{model_name}.yaml"),
        device="cuda:0",
        is_half=False,
    )
    for index, item in enumerate(manifest["inputs"], 1):
        raw = Path(item["raw"])
        vocal = Path(item["vocals"])
        other = Path(item["other"])
        if vocal.exists() and other.exists():
            print(f"[{index}/{len(INPUTS)}] already separated {raw.name}", flush=True)
            continue
        print(f"[{index}/{len(INPUTS)}] separate {raw.name}", flush=True)
        loader._path_audio_(str(raw), str(OTHER), str(VOCALS), "wav", False)
        if not vocal.exists() or not other.exists():
            raise RuntimeError(f"Separation did not create expected stems for {raw}")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
