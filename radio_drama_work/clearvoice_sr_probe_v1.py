"""Run a small ClearVoice SR A/B probe on already high-confidence raw clips.

This is deliberately a probe, not a training export. Super-resolution can
make a file brighter while changing speaker identity, so every output is
kept separate until the identity/noise gate is evaluated.
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
WORK = ROOT / "clearvoice_sr_probe_v1"
MODEL_WORK = ROOT / "clearvoice_sr_smoke_v1"
INPUT_DIR = ROOT / "full_funasr_gold_v9" / "accepted_segments"
WORK.mkdir(parents=True, exist_ok=True)
CHECKPOINT = MODEL_WORK / "checkpoints" / "MossFormer2_SR_48K"
REQUIRED = [CHECKPOINT / "last_best_checkpoint_m.pt", CHECKPOINT / "last_best_checkpoint_g.pt"]
if not all(path.exists() and path.stat().st_size > 10_000_000 for path in REQUIRED):
    raise FileNotFoundError(f"incomplete ClearVoice checkpoint: {REQUIRED}")
# ClearVoice resolves its checkpoint path relative to cwd. Keep inference cwd
# on the verified local model directory; never allow a failed download to run.
os.chdir(MODEL_WORK)


def main() -> None:
    inputs = sorted(INPUT_DIR.glob("*.wav"))[:5]
    if not inputs:
        raise FileNotFoundError(INPUT_DIR)
    print(f"inputs={len(inputs)}", flush=True)
    model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
    rows = []
    for index, input_path in enumerate(inputs):
        output_path = WORK / f"sr_{index:02d}_{input_path.stem}.wav"
        print(f"processing={index + 1}/{len(inputs)} {input_path.name}", flush=True)
        output = model(input_path=str(input_path), online_write=False)
        model.write(output, output_path=str(output_path))
        info = sf.info(str(output_path))
        rows.append({
            "input": str(input_path),
            "output": str(output_path),
            "sample_rate": info.samplerate,
            "duration_sec": info.duration,
            "frames": info.frames,
        })
        print(f"saved={output_path.name} sr={info.samplerate} duration={info.duration:.3f}", flush=True)
    (WORK / "probe_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={WORK / 'probe_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
