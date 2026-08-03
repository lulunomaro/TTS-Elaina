# -*- coding: utf-8 -*-
"""Run ClearVoice exactly like the official Numpy2Numpy separation demo."""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, r"radio_drama_work/third_party/clearvoice_pkg")
from clearvoice import ClearVoice  # noqa: E402

ROOT = Path(r"radio_drama_work/radio_drama_work")
INPUT_DIR = ROOT / "clearvoice_test" / "inputs_windows"
OUTPUT_DIR = ROOT / "clearvoice_test" / "outputs_official_numpy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

my_clearvoice = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
num_spks = 2

for input_path in sorted(INPUT_DIR.glob("*.wav")):
    audio, sr = sf.read(str(input_path))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
        audio = np.reshape(audio, [1, audio.shape[0]])
    audio = audio.astype(np.float32)
    output_wav = my_clearvoice(audio, False)
    for spk in range(num_spks):
        output_path = OUTPUT_DIR / f"{input_path.stem}_spk{spk + 1}.wav"
        sf.write(str(output_path), output_wav[spk, 0, :], 16000)
        print(output_path)
