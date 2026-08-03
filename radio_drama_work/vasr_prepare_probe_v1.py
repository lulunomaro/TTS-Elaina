from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
sys.path.insert(0, str(ROOT / "vasr_light_shims"))
sys.path.append(str(ROOT / "vasr_deps"))
sys.path.append(r"radio_drama_work/versatile_audio_super_resolution-main/versatile_audio_super_resolution-main")
import transformers
transformers.RobertaTokenizer = type("DummyTokenizer", (), {"from_pretrained": classmethod(lambda cls, *a, **k: cls())})
transformers.RobertaConfig.from_pretrained = classmethod(lambda cls, *a, **k: cls(max_position_embeddings=514, type_vocab_size=1))
from audiosr.pipeline import make_batch_for_super_resolution

lines = [x for x in (ROOT / "full_funasr_gold_train_candidate_v9.list").read_text(encoding="utf-8").splitlines() if x.strip()]
for index in (0, 1, 2, 3, 4, 5):
    path = Path(lines[index].split("|", 1)[0])
    started = time.perf_counter()
    print("start", index, path.name, flush=True)
    batch, duration = make_batch_for_super_resolution(str(path))
    print("done", index, duration, time.perf_counter() - started, batch["waveform"].shape, flush=True)
