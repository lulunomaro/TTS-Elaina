"""Run verified ClearVoice SR over the v9 high-confidence raw clips.

Outputs are an isolated repair tier. They are not merged into the raw list
until the identity gate passes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import soundfile as sf

for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, r"radio_drama_work/third_party/clearvoice_pkg")
from clearvoice import ClearVoice

ROOT = Path(r"radio_drama_work/radio_drama_work")
MODEL_WORK = ROOT / "clearvoice_sr_smoke_v1"
WORK = ROOT / "clearvoice_sr_full_v1"
INPUT_LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"
CHECKPOINT = MODEL_WORK / "checkpoints" / "MossFormer2_SR_48K"
REQUIRED = [CHECKPOINT / "last_best_checkpoint_m.pt", CHECKPOINT / "last_best_checkpoint_g.pt"]


def main() -> None:
    if not all(path.exists() and path.stat().st_size > 10_000_000 for path in REQUIRED):
        raise FileNotFoundError(f"incomplete ClearVoice checkpoint: {REQUIRED}")
    inputs = []
    for line in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            path = Path(line.split("|", 1)[0].strip())
            if path.exists():
                inputs.append(path)
    if not inputs:
        raise FileNotFoundError(INPUT_LIST)
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(MODEL_WORK)
    print(f"inputs={len(inputs)}", flush=True)
    model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
    rows = []
    for index, input_path in enumerate(inputs):
        output_path = WORK / f"sr_{index:05d}_{input_path.stem}.wav"
        print(f"processing={index + 1}/{len(inputs)} {input_path.name}", flush=True)
        output = model(input_path=str(input_path), online_write=False)
        model.write(output, output_path=str(output_path))
        info = sf.info(str(output_path))
        rows.append({"input": str(input_path), "output": str(output_path), "sample_rate": info.samplerate, "duration_sec": info.duration, "frames": info.frames})
    (WORK / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(WORK / 'manifest.json'), "count": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
