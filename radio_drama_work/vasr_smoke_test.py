from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"radio_drama_work/hf_cache")
# The desktop proxy is optional. Set VASR_USE_PROXY=1 when the worker can
# reach the user's local 7897 proxy; otherwise avoid stale proxy variables.
if os.environ.get("VASR_USE_PROXY") != "1":
    for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(_proxy_key, None)

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, r"radio_drama_work/vasr_light_shims")
sys.path.append(r"radio_drama_work/vasr_deps")
sys.path.append(r"radio_drama_work/versatile_audio_super_resolution-main/versatile_audio_super_resolution-main")

import transformers


class DummyRobertaTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


transformers.RobertaTokenizer = DummyRobertaTokenizer
_RobertaConfig = transformers.RobertaConfig
_RobertaConfig.from_pretrained = classmethod(lambda cls, *args, **kwargs: cls(max_position_embeddings=514, type_vocab_size=1))

from audiosr import build_model, super_resolution

ROOT = Path(r"radio_drama_work/radio_drama_work")
INPUT = ROOT / "full_funasr_gold_v9" / "accepted_segments" / "gold_00000_243358894-1-192_vocals_0002232960_0002291840.wav"
OUT = ROOT / "vasr_smoke_v1"
CKPT = ROOT / "hf_cache" / "hub" / "models--haoheliu--audiosr_speech" / "blobs" / "59fd60a9b93346c698788bac9b6077977a21733e7a78db9d3c3af3f1bdebc640"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if not CKPT.exists() or CKPT.stat().st_size != 6178193436:
        raise FileNotFoundError(f"incomplete VASR speech checkpoint: {CKPT}")
    print(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    print("loading AudioSR speech model", flush=True)
    import audiosr.pipeline as audiosr_pipeline
    audiosr_pipeline.download_checkpoint = lambda checkpoint_name="speech": str(CKPT)
    model = build_model(model_name="speech", device="cuda:0" if torch.cuda.is_available() else "cpu")
    print("model_loaded", flush=True)
    result = super_resolution(model, str(INPUT), seed=42, guidance_scale=3.5, ddim_steps=20, latent_t_per_second=12.8)
    array = result[0] if hasattr(result, "__len__") else result
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32).squeeze()
    path = OUT / f"vasr_speech_{INPUT.stem}.wav"
    sf.write(str(path), array, 48000, subtype="PCM_16")
    print(f"saved={path} samples={len(array)} duration={len(array)/48000:.3f}", flush=True)


if __name__ == "__main__":
    main()
