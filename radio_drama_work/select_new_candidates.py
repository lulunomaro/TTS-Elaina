import json
import os
import re

score_path = r"radio_drama_work/new_speaker_scores.json"
asr_path = r"radio_drama_work/new_candidates_asr/new_candidates_asr.json"
out_dir = r"radio_drama_work/new_curated"
scores = json.load(open(score_path, encoding="utf-8"))["cand_scores"]
rows = json.load(open(asr_path, encoding="utf-8"))

reject_terms = ("イレイナさん", "イレーナさん")
selected = []
rejected = []
for r in rows:
    name = r["name"]
    score = scores.get(name, -1.0)
    text = (r.get("text") or "").strip()
    a, b = name.rsplit("_", 2)[-2:]
    dur = (int(b[:-4]) - int(a)) / 32000
    reasons = []
    if score < 0.76:
        reasons.append("speaker<0.76")
    if r.get("avg_logprob", -99) < -0.45:
        reasons.append("asr<-.45")
    if len(text) < 5:
        reasons.append("text<5")
    if len(text) > 58:
        reasons.append("text>58")
    if dur < 2.0 or dur > 8.0:
        reasons.append("duration")
    if any(term in text for term in reject_terms):
        reasons.append("other-address")
    if reasons:
        rejected.append((r, reasons))
    else:
        selected.append(r)

selected.sort(key=lambda r: (-scores[r["name"]], r["name"]))
print("selected", len(selected), "rejected", len(rejected))
for r in selected:
    print(f"KEEP/t{scores[r['name']]:.3f}/t{r['name']}/t{r['text']}")
print("/nREJECTS")
for r, reasons in sorted(rejected, key=lambda x: (-scores[x[0]["name"]], x[0]["name"])):
    print(f"DROP/t{scores[r['name']]:.3f}/t{','.join(reasons)}/t{r['name']}/t{r.get('text','')}")

os.makedirs(out_dir, exist_ok=True)
list_path = os.path.join(out_dir, "new_curated.list")
with open(list_path, "w", encoding="utf-8") as f:
    for r in selected:
        f.write(f"{r['file']}|new_radio|JA|{r['text']}/n")
with open(os.path.join(out_dir, "new_curated.json"), "w", encoding="utf-8") as f:
    json.dump({"selected": selected, "rejected": [{"row": r, "reasons": why} for r, why in rejected]}, f, ensure_ascii=False, indent=2)
print("saved", list_path)
