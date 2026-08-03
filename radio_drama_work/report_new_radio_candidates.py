import json
sv = {r['name']: r for r in json.load(open('/root/autodl-tmp/ireina_new_materials_v1/sv_scores.json'))}
asr = json.load(open('/root/autodl-tmp/ireina_new_materials_v1/asr/asr.json'))
rows=[]
for r in asr:
    s=sv[r['name']]
    row={**r, **{k:s[k] for k in ('best_score','median','fraction_ge_075','fraction_ge_080','best_start','best_end')}}
    rows.append(row)
print('ASR rows',len(rows),'with_text',sum(bool(r.get('text')) for r in rows))
for cond_name, cond in [
    ('strict', lambda r: r['median']>=.75 and r['avg_logprob']>=-.45 and len(r.get('text',''))>=3),
    ('speaker_strict', lambda r: r['median']>=.70 and r['fraction_ge_075']>=.5 and r['avg_logprob']>=-.55 and len(r.get('text',''))>=3),
    ('usable', lambda r: r['median']>=.65 and r['best_score']>=.72 and r['fraction_ge_075']>=.33 and r['avg_logprob']>=-.60 and len(r.get('text',''))>=3),
]:
    hit=[r for r in rows if cond(r)]
    print('/n',cond_name,len(hit))
    for r in sorted(hit,key=lambda r:(r['median'],r['best_score']),reverse=True):
        print(f"{r['median']:.3f}/t{r['best_score']:.3f}/t{r['fraction_ge_075']:.2f}/t{r['avg_logprob']:.3f}/t{r['name']}/t{r.get('text','')}")
