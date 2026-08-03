import json
import os
import sys
from pathlib import Path

from faster_whisper import WhisperModel


inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"radio_drama_work/speaker_aware_candidates/review")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.parent / "speaker_aware_review_asr.json"
model_path = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50/tools/asr/models/faster-whisper-large-v3-turbo")
model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=8)
rows = []
for path in sorted(inp.glob("*.wav")):
    segs, info = model.transcribe(
        str(path), language="ja", beam_size=5, temperature=0.0,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False, compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6,
    )
    segs = list(segs)
    text = "".join(s.text for s in segs).strip()
    lp = sum(float(s.avg_logprob) for s in segs) / len(segs) if segs else -99.0
    row = {"file": str(path), "name": path.name, "text": text,
           "avg_logprob": lp, "segments": len(segs),
           "language_probability": float(info.language_probability)}
    rows.append(row)
    print(f"{path.name}/t{lp:.3f}/t{text}", flush=True)
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved={out} rows={len(rows)}")
