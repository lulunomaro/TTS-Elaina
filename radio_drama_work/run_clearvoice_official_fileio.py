# -*- coding: utf-8 -*-
"""Run the single-file ClearVoice flow exactly as demo.py documents it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"radio_drama_work/third_party/clearvoice_pkg")
from clearvoice import ClearVoice  # noqa: E402

ROOT = Path(r"radio_drama_work/radio_drama_work")
INPUT_DIR = ROOT / "clearvoice_test" / "inputs_windows"
OUTPUT_DIR = ROOT / "clearvoice_test" / "outputs_official_fileio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

my_clearvoice = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
for input_path in sorted(INPUT_DIR.glob("*.wav")):
    output_wav = my_clearvoice(input_path=str(input_path), online_write=False)
    output_path = OUTPUT_DIR / f"{input_path.stem}.wav"
    my_clearvoice.write(output_wav, output_path=str(output_path))
    print(output_path)
