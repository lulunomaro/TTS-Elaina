#!/usr/bin/env python3
"""Build the paired manifest in-place from the server's existing D_old list."""
import json
from pathlib import Path

ROWS = [14, 15, 18, 22, 42, 46, 54, 68, 73, 74, 80, 81, 83, 86, 88, 91, 96, 107, 124, 146, 152, 154]
SCENES = {
    14: "surprise", 15: "surprise", 18: "anger", 22: "dialogue",
    42: "cold", 46: "taste_reaction", 54: "hesitation", 68: "sadness",
    73: "anger", 74: "confusion", 80: "anticipation", 81: "question",
    83: "tottemo_ending", 86: "surprise_anger", 88: "chant",
    91: "fatigue", 96: "laugh", 107: "sigh", 124: "pause",
    146: "cry", 152: "hesitation", 154: "dark_emotion",
}
SOURCE = Path("/root/autodl-tmp/ireina_v2/data/D_old.list")
OUT = Path("/root/autodl-tmp/paired_emo_eval.jsonl")
lines = [line for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = []
for index in ROWS:
    path, speaker, lang, text = lines[index - 1].split("|", 3)
    rows.append({"sample_id": f"r{index:03d}", "row": index, "real_path": path,
                 "speaker": speaker, "lang": lang, "scene": SCENES[index], "text": text})
OUT.write_text("/n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "/n", encoding="utf-8")
print(f"wrote={OUT} rows={len(rows)}")
print(next(row for row in rows if row["row"] == 83)["text"])
