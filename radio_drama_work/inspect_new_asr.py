import json
import statistics

path = r"radio_drama_work/new_candidates_asr/new_candidates_asr.json"
rows = json.load(open(path, encoding="utf-8"))
ok = [r for r in rows if r.get("text") and not r.get("error")]
print("rows", len(rows), "with_text", len(ok))
langs = sorted(set(r.get("language") for r in rows))
print("languages", {x: sum(r.get("language") == x for r in rows) for x in langs})
print("text_len", min(map(lambda r: len(r["text"]), ok)), statistics.median(map(lambda r: len(r["text"]), ok)), max(map(lambda r: len(r["text"]), ok)))
print("median_avg_logprob", statistics.median(r["avg_logprob"] for r in ok))
print("median_max_no_speech_prob", statistics.median(r["max_no_speech_prob"] for r in ok))
print("/nlowest no-speech probability / likely speech:")
for r in sorted(ok, key=lambda r: (r["max_no_speech_prob"], -r["avg_logprob"]))[:20]:
    print(f"{r['name']}/t{r['avg_logprob']:.3f}/t{r['max_no_speech_prob']:.3f}/t{r['text']}")
print("/nlowest log probability:")
for r in sorted(ok, key=lambda r: (r["avg_logprob"], -len(r["text"])))[:20]:
    print(f"{r['name']}/t{r['avg_logprob']:.3f}/t{r['max_no_speech_prob']:.3f}/t{r['text']}")
