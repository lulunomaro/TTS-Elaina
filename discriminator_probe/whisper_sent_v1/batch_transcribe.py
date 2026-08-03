# -*- coding: utf-8 -*-
"""Batch transcribe all 19 sources with faster-whisper large-v3-turbo (sentence-level).
Resumable: skips sources already transcribed. Run in background."""
import io, json, os, sys, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

ROOT = r"F:\GPTSOVITS\radio_drama_work"
OUT = r"F:\GPTSOVITS\discriminator_probe\whisper_sent_v1"
os.makedirs(OUT, exist_ok=True)

anime_manifest = json.loads(io.open(os.path.join(ROOT, "anime_batch_v1", "manifest.json"), encoding="utf-8").read())
radio_manifest = json.loads(io.open(os.path.join(ROOT, "full_track_manifest_v1.json"), encoding="utf-8").read())

sources = []
for t in anime_manifest.get("tracks", []):
    if t["id"].startswith("majo_"):
        sources.append((t["id"], t["vocals"]))
for t in radio_manifest.get("tracks", []):
    sources.append((t["id"], t["vocals"]))
print(f"total sources: {len(sources)}", flush=True)

log = io.open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")
def log_write(msg):
    log.write(time.strftime("%H:%M:%S ") + msg + "\n")
    log.flush()
    print(msg, flush=True)

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
log_write("model loaded (cuda float16)")

for sid, vocals in sources:
    dest = os.path.join(OUT, sid + ".json")
    if os.path.exists(dest):
        log_write(f"skip {sid} (exists)")
        continue
    if not os.path.exists(vocals):
        log_write(f"MISSING {sid} {vocals}")
        continue
    t0 = time.time()
    audio = decode_audio(vocals, sampling_rate=16000)
    segs, _ = model.transcribe(audio, beam_size=5, language="ja", vad_filter=True,
                               condition_on_previous_text=False)
    items = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
             for s in segs]
    with io.open(dest, "w", encoding="utf-8") as f:
        json.dump({"source": sid, "segments": items}, f, ensure_ascii=False)
    dur = len(audio) / 16000
    log_write(f"done {sid} dur={dur:.0f}s segs={len(items)} elapsed={time.time()-t0:.0f}s")

log_write("ALL DONE")
