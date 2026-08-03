#!/bin/bash
set -euo pipefail
PY=/root/miniconda3/envs/gptsovits/bin/python
IN=/root/autodl-tmp/new_candidates_sp75
OUT=/root/autodl-tmp/new_candidates_asr
MODEL=/root/autodl-tmp/pkg/GPT-SoVITS-v2pro-20250604-nvidia50/tools/asr/models/faster-whisper-large-v3-turbo
mkdir -p "$OUT"
"$PY" - "$IN" "$OUT" "$MODEL" <<'PY'
import json, os, sys
from faster_whisper import WhisperModel

inp, out, model_path = sys.argv[1:]
# The server's torch CUDA is available, but CTranslate2 lacks libcublas.so.12.
# Use CPU/int8 for this small quality-gate batch rather than mutating the env.
model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=8)
rows = []
for name in sorted(os.listdir(inp)):
    if not name.lower().endswith(".wav"):
        continue
    path = os.path.join(inp, name)
    try:
        segments, info = model.transcribe(
            path, language="ja", beam_size=5, temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()
        avg_lp = sum(float(s.avg_logprob) for s in segs) / len(segs) if segs else -99.0
        max_ns = max((float(s.no_speech_prob) for s in segs), default=1.0)
        dur = max((float(s.end) for s in segs), default=0.0)
        row = dict(file=path, name=name, language=info.language,
                   language_probability=float(info.language_probability),
                   text=text, segments=len(segs), avg_logprob=avg_lp,
                   max_no_speech_prob=max_ns, asr_duration=dur)
        rows.append(row)
        print(f"{name}/t{info.language}/t{info.language_probability:.3f}/t{avg_lp:.3f}/t{max_ns:.3f}/t{text}", flush=True)
    except Exception as e:
        rows.append(dict(file=path, name=name, error=repr(e), text=""))
        print(f"FAIL/t{name}/t{e}", flush=True)

with open(os.path.join(out, "new_candidates_asr.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
with open(os.path.join(out, "new_candidates_asr.list"), "w", encoding="utf-8") as f:
    for r in rows:
        if r.get("text"):
            f.write(f"{r['file']}|new_radio|JA|{r['text']}/n")
print(f"saved {len(rows)} rows")
PY
