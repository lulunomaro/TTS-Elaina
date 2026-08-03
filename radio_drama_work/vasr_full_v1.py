"""Batch AudioSR speech repair on v9 raw clips, cropped back to source length."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HOME", r"radio_drama_work/hf_cache")
sys.path.insert(0, r"radio_drama_work/vasr_light_shims")
sys.path.append(r"radio_drama_work/vasr_deps")
sys.path.append(r"radio_drama_work/versatile_audio_super_resolution-main/versatile_audio_super_resolution-main")

import transformers


class DummyRobertaTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


transformers.RobertaTokenizer = DummyRobertaTokenizer
transformers.RobertaConfig.from_pretrained = classmethod(lambda cls, *args, **kwargs: cls(max_position_embeddings=514, type_vocab_size=1))

from audiosr import build_model, super_resolution

ROOT = Path(r"radio_drama_work/radio_drama_work")
INPUT_LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"
WORK = ROOT / os.environ.get("VASR_WORK", "vasr_full_v1")
OUT_DIR = WORK / "cropped_32k"
CKPT = ROOT / "hf_cache" / "hub" / "models--haoheliu--audiosr_speech" / "blobs" / "59fd60a9b93346c698788bac9b6077977a21733e7a78db9d3c3af3f1bdebc640"


def main() -> None:
    if not CKPT.exists() or CKPT.stat().st_size != 6178193436:
        raise FileNotFoundError(CKPT)
    inputs = [Path(line.split("|", 1)[0].strip()) for line in INPUT_LIST.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    start = int(os.environ.get("VASR_START", "0"))
    stop = int(os.environ.get("VASR_STOP", str(len(inputs))))
    inputs = inputs[start:stop]
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    padded_dir = WORK / "padded_inputs"
    padded_dir.mkdir(parents=True, exist_ok=True)
    pad_sec = float(os.environ.get("VASR_PAD_SEC", "0"))
    ddim_steps = int(os.environ.get("VASR_DDIM_STEPS", "20"))
    input_sr = int(os.environ.get("VASR_INPUT_SR", "0"))
    import audiosr.pipeline as audiosr_pipeline
    audiosr_pipeline.download_checkpoint = lambda checkpoint_name="speech": str(CKPT)
    # The source is already a BS-Roformer vocal stem. After resampling a
    # 32kHz source to 48kHz it is intrinsically band-limited at ~16kHz, so
    # avoid AudioSR's random short-signal scipy low-pass branch. That branch
    # can become extremely slow for a few short clips and is unnecessary here.
    def stable_lowpass(batch):
        return {"waveform_lowpass": batch["waveform"].clone()}
    audiosr_pipeline.lowpass_filtering_prepare_inference = stable_lowpass
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"inputs={len(inputs)} device={device}", flush=True)
    model = build_model(model_name="speech", device=device)
    rows = []
    for index, input_path in enumerate(inputs):
        raw, sr = sf.read(str(input_path), always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        duration = len(raw) / sr
        final_path = OUT_DIR / f"vasr_{index:05d}_{input_path.stem}.wav"
        try:
            print(f"processing={index + 1}/{len(inputs)} {input_path.name}", flush=True)
            inference_path = input_path
            if pad_sec > 0 or input_sr > 0:
                prepared = raw.astype(np.float32)
                prepared_sr = sr
                if input_sr > 0 and input_sr != sr:
                    prepared = librosa.resample(prepared, orig_sr=sr, target_sr=input_sr).astype(np.float32)
                    prepared_sr = input_sr
                padded = np.pad(prepared, (0, round(pad_sec * prepared_sr)))
                inference_path = padded_dir / input_path.name
                sf.write(str(inference_path), padded, prepared_sr, subtype="PCM_16")
            result = super_resolution(model, str(inference_path), seed=42, guidance_scale=3.5, ddim_steps=ddim_steps, latent_t_per_second=12.8)
            array = result[0] if hasattr(result, "__len__") else result
            if torch.is_tensor(array):
                array = array.detach().cpu().numpy()
            array = np.asarray(array, dtype=np.float32).squeeze()
            target_frames_48k = round(duration * 48000)
            if len(array) < target_frames_48k:
                raise RuntimeError(f"generated audio shorter than source: {len(array)} < {target_frames_48k}")
            array = array[:target_frames_48k]
            final = librosa.resample(array, orig_sr=48000, target_sr=32000).astype(np.float32)
            final = final[:round(duration * 32000)]
            sf.write(str(final_path), final, 32000, subtype="PCM_16")
            rows.append({"input": str(input_path), "output": str(final_path), "sample_rate": 32000, "duration_sec": len(final) / 32000, "status": "ok"})
            print(f"saved={final_path.name} duration={len(final)/32000:.3f}", flush=True)
        except Exception as exc:
            rows.append({"input": str(input_path), "output": str(final_path), "status": "error", "error": repr(exc)})
            print(f"error={input_path.name} {exc!r}", flush=True)
    (WORK / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(WORK / 'manifest.json'), "ok": sum(x['status'] == 'ok' for x in rows), "errors": sum(x['status'] != 'ok' for x in rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
