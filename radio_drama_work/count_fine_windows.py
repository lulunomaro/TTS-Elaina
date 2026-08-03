import json
from collections import Counter

rows = json.load(open(r"radio_drama_work/new_candidates_fine_sv.json", encoding="utf-8"))
for th in (.72, .74, .75, .76, .78, .80):
    vals = [(r["name"], w) for r in rows for w in r["windows"] if w["score"] >= th]
    print(th, len(vals), Counter(n.split("_vocals")[0] for n, _ in vals))
print("top windows")
for r in rows:
    for w in r["windows"]:
        if w["score"] >= .78:
            print(f"{w['score']:.3f}/t{w['start']:.2f}-{w['end']:.2f}/t{r['name']}")
