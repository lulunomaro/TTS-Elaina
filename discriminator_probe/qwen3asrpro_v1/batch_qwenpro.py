# -*- coding: utf-8 -*-
"""Full-source reprocess with Qwen3ASRpro combo: pyannote community-1 diarization
+ Qwen3-ASR-1.7B chunked transcribe + ForcedAligner char timestamps.
Output: per-source json {segments:[{start,end,text,speaker}]}. Resumable."""
import io
import json
import os
import sys
import time
from collections import defaultdict

BASE = r"F:\GPTSOVITS\qwen3asr_pro\Qwen3_ASR"
sys.path.insert(0, os.path.join(BASE, "app"))
os.environ["PATH"] = os.path.join(BASE, "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from model_hub import ModelHub
from audio_utils import AudioUtils

ROOT = r"F:\GPTSOVITS\radio_drama_work"
OUT = r"F:\GPTSOVITS\discriminator_probe\qwen3asrpro_v1"
os.makedirs(OUT, exist_ok=True)
log = io.open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

def logw(msg):
    log.write(time.strftime("%H:%M:%S ") + msg + "\n")
    log.flush()
    print(msg, flush=True)

sources = []
for t in json.loads(io.open(os.path.join(ROOT, "anime_batch_v1", "manifest.json"), encoding="utf-8").read()).get("tracks", []):
    if t["id"].startswith("majo_"):
        sources.append((t["id"], t["vocals"]))
for t in json.loads(io.open(os.path.join(ROOT, "full_track_manifest_v1.json"), encoding="utf-8").read()).get("tracks", []):
    sources.append((t["id"], t["vocals"]))
logw(f"sources: {len(sources)}")

hub = ModelHub()
t0 = time.time()
hub.load_diarization(os.path.join(BASE, "models", "speaker-diarization-community-1"))
hub.load_asr(
    os.path.join(BASE, "models", "Qwen", "Qwen3-ASR-1.7B"), "transformers",
    {"dtype": torch.bfloat16, "device_map": "cuda:0", "max_inference_batch_size": 2, "max_new_tokens": 256},
    aligner_ckpt=os.path.join(BASE, "models", "Qwen", "Qwen3-ForcedAligner-0.6B"),
    aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
)
logw("models loaded %d s" % int(time.time() - t0))
from transcribe_engine import TranscribeEngine
engine = TranscribeEngine(hub)

# ForcedAligner supports up to 5min speech per call -> chunk at 240s for full alignment
import audio_utils
_orig_split = audio_utils.AudioUtils.split_audio

def split_audio(input_path, max_sec=240):
    return _orig_split(input_path, max_sec=max_sec)

audio_utils.AudioUtils.split_audio = staticmethod(split_audio)

for sid, vocals in sources:
    dest = os.path.join(OUT, sid + ".json")
    if os.path.exists(dest):
        logw(f"skip {sid} (exists)")
        continue
    t0 = time.time()
    try:
        full_text, timestamps, sd_res = engine.run(vocals, "Japanese", True, "", "", "")
        items = [{"start": round(ts["start_time"], 3), "end": round(ts["end_time"], 3), "text": ts["text"]}
                 for ts in timestamps]
        spk = [[round(s[0], 3), round(s[1], 3), s[2]] for s in (sd_res["text"] if sd_res else [])]
        with io.open(dest, "w", encoding="utf-8") as f:
            json.dump({"source": sid, "chars": items, "diarization": spk, "fulltext": full_text}, f, ensure_ascii=False)
        logw(f"done {sid} chars={len(items)} spk={len(spk)} elapsed={time.time()-t0:.0f}s")
    except Exception as e:
        logw(f"FAIL {sid}: {str(e)[:200]}")
        import traceback
        log.write(traceback.format_exc()[-800:] + "\n")
        log.flush()
logw("ALL DONE")
