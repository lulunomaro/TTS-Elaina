from __future__ import annotations

import sys

# Keep the bundled CUDA torch/torchaudio pair first.  The isolated dependency
# folder supplies torchvision/timm/etc. without replacing that torch build.
import torch
import numpy
sys.path.insert(0, r"radio_drama_work/vasr_light_shims")
sys.path.append(r"radio_drama_work/vasr_deps")
sys.path.append(r"radio_drama_work/versatile_audio_super_resolution-main/versatile_audio_super_resolution-main")

import transformers


class DummyRobertaTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


transformers.RobertaTokenizer = DummyRobertaTokenizer

import audiosr
from audiosr import build_model

print("audiosr_ok", audiosr.__file__)
print("build_model_ok", build_model)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__)
