from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, r"radio_drama_work/third_party/clearvoice_pkg")
from clearvoice import ClearVoice

ROOT = Path(r"radio_drama_work/radio_drama_work")
WORK = ROOT / "clearvoice_sr_smoke_v1"
INPUT = ROOT / "full_funasr_gold_v9" / "accepted_segments" / "gold_00000_243358894-1-192_vocals_0002232960_0002291840.wav"
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)


def main() -> None:
    print(f"input={INPUT}", flush=True)
    model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
    path = WORK / "clearvoice_MossFormer2_SR_48K.wav"
    output = model(input_path=str(INPUT), online_write=False)
    print(f"result_type={type(output)}", flush=True)
    model.write(output, output_path=str(path))
    info = sf.info(str(path))
    print(f"saved={path} samples={info.frames} sr={info.samplerate} duration={info.duration:.3f}", flush=True)


if __name__ == "__main__":
    main()
