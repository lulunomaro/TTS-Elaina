#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/gptsovits/bin/python
IN=/root/autodl-tmp/new_candidates_sp75
FINE=/root/autodl-tmp/new_candidates_fine_sv.json
OUT=/root/autodl-tmp/new_fine_windows_v2
ASROUT=/root/autodl-tmp/new_fine_windows_asr_v2
MODEL=/root/autodl-tmp/pkg/GPT-SoVITS-v2pro-20250604-nvidia50/tools/asr/models/faster-whisper-large-v3-turbo
rm -rf "$OUT" "$ASROUT"
mkdir -p "$OUT" "$ASROUT"
"$PY" - "$IN" "$FINE" "$OUT" <<'PY'
import json, os, sys, wave
import numpy as np
import torchaudio

inp, fine_path, out = sys.argv[1:]
rows = json.load(open(fine_path, encoding="utf-8"))
meta = []
for idx, r in enumerate(rows):
    if r["best_score"] < 0.76:
        continue
    # one best, non-overlapping window per source slice keeps the set diverse
    w = max(r["windows"], key=lambda x: x["score"])
    src = os.path.join(inp, r["name"])
    wav, sr = torchaudio.load(src)
    st, en = int(w["start"] * sr), int(w["end"] * sr)
    clip = wav[:, st:min(en, wav.shape[1])]
    name = f"fine_{idx:03d}_{r['best_score']:.3f}_{r['name']}"
    dst = os.path.join(out, name)
    pcm = (clip.clamp(-1, 1).numpy().T * 32767).astype(np.int16)
    with wave.open(dst, "wb") as wf:
        wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    meta.append(dict(file=dst, source=r["name"], score=r["best_score"], start=w["start"], end=w["end"]))
with open(os.path.join(out, "windows_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("extracted", len(meta))
PY
"$PY" - "$OUT" "$ASROUT" "$MODEL" <<'PY'
import json, os, sys
from faster_whisper import WhisperModel

inp, out, model_path = sys.argv[1:]
model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=8)
meta = {r["file"]: r for r in json.load(open(os.path.join(inp, "windows_meta.json"), encoding="utf-8"))}
rows = []
for name in sorted(os.listdir(inp)):
    if not name.lower().endswith(".wav"): continue
    path = os.path.join(inp, name)
    segs, info = model.transcribe(path, language="ja", beam_size=5, temperature=0.0,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=350),
        condition_on_previous_text=False, compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6)
    segs = list(segs)
    text = "".join(s.text for s in segs).strip()
    lp = sum(float(s.avg_logprob) for s in segs) / len(segs) if segs else -99.0
    r = dict(file=path, name=name, text=text, avg_logprob=lp,
             source=meta[path]["source"], score=meta[path]["score"],
             start=meta[path]["start"], end=meta[path]["end"])
    rows.append(r)
    print(f"{name}/t{r['score']:.3f}/t{lp:.3f}/t{text}", flush=True)
with open(os.path.join(out, "new_fine_windows_asr.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
with open(os.path.join(out, "new_fine_windows_asr.list"), "w", encoding="utf-8") as f:
    for r in rows:
        if r["text"] and r["avg_logprob"] >= -.55 and len(r["text"]) >= 3:
            f.write(f"{r['file']}|new_radio|JA|{r['text']}/n")
print("saved", len(rows))
PY
