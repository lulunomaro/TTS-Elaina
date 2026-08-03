import json
import os
from collections import Counter

src = r"radio_drama_work/new_fine_windows_v2/new_fine_windows_asr.json"
out_dir = r"radio_drama_work/new_fine_windows_v2"
rows = json.load(open(src, encoding="utf-8"))
bad_terms = ("イレイナさん", "イレーナさん", "何か言ってる", "悪人の下っ端")
confirmed_non_targets = {
    "28055832533-1-192_vocals_0017516160_0017573120.wav",
}
keep, drop = [], []
for r in rows:
    why = []
    if r.get("name") in confirmed_non_targets or r.get("source") in confirmed_non_targets:
        why.append("confirmed_non_target")
    if r.get("score", 0) < .76: why.append("fine_score")
    if r.get("avg_logprob", -99) < -.50: why.append("asr_conf")
    if len(r.get("text", "")) < 5: why.append("short_text")
    if any(x in r.get("text", "") for x in bad_terms): why.append("suspect_text")
    (drop if why else keep).append((r, why))
print("keep", len(keep), Counter(x[0]["source"].split("_vocals")[0] for x in keep))
for r, _ in keep: print(f"KEEP/t{r['score']:.3f}/t{r['avg_logprob']:.3f}/t{r['name']}/t{r['text']}")
print("drop", len(drop))
for r, why in drop: print(f"DROP/t{','.join(why)}/t{r['name']}/t{r['text']}")
list_path = os.path.join(out_dir, "new_fine_curated.list")
with open(list_path, "w", encoding="utf-8") as f:
    for r, _ in keep:
        f.write(f"{r['file']}|new_radio|JA|{r['text']}/n")
with open(os.path.join(out_dir, "new_fine_curated.json"), "w", encoding="utf-8") as f:
    json.dump({"selected": [r for r, _ in keep], "dropped": [{"row": r, "reasons": why} for r, why in drop]}, f, ensure_ascii=False, indent=2)
print("saved", list_path)
