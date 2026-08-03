import json
import collections

scores = json.load(open(r"radio_drama_work/new_speaker_scores.json", encoding="utf-8"))["cand_scores"]
asr = {r["name"]: r for r in json.load(open(r"radio_drama_work/new_candidates_asr/new_candidates_asr.json", encoding="utf-8"))}
fine = {r["name"]: r for r in json.load(open(r"radio_drama_work/new_candidates_fine_sv.json", encoding="utf-8"))}
for threshold in [(0.78, .74, .50), (0.76, .72, .33), (0.80, .75, .50), (0.78, .75, .50)]:
    b, med, frac = threshold
    keep = []
    for name, f in fine.items():
        r = asr[name]
        text = r.get("text", "")
        if f["best_score"] >= b and (f["median"] >= med or f["fraction_ge_075"] >= frac) and len(text) >= 5 and len(text) <= 58 and r.get("avg_logprob", -99) >= -.45:
            keep.append(name)
    print(threshold, len(keep), dict(collections.Counter(n.split("_vocals")[0] for n in keep)))
    for n in sorted(keep, key=lambda x: -scores[x]):
        print(f"  {scores[n]:.3f} fine={fine[n]['best_score']:.3f}/{fine[n]['median']:.3f}/{fine[n]['fraction_ge_075']:.2f}/t{n}/t{asr[n]['text']}")
