import json
import os

score_path = r"radio_drama_work/new_speaker_scores.json"
asr_path = r"radio_drama_work/new_candidates_asr/new_candidates_asr.json"
scores = json.load(open(score_path, encoding="utf-8"))["cand_scores"]
rows = json.load(open(asr_path, encoding="utf-8"))
for r in sorted(rows, key=lambda r: (-scores.get(r["name"], -1), r["name"])):
    name = r["name"]
    start, end = name.rsplit("_", 2)[-2:]
    dur = (int(end[:-4]) - int(start)) / 32000
    print(f"{scores.get(name, -1):.3f}/t{dur:4.1f}s/t{r.get('avg_logprob', -99):.3f}/t{name}/t{r.get('text','')}")
