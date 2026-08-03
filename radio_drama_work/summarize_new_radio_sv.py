import json, statistics
rows = json.load(open('/root/autodl-tmp/ireina_new_materials_v1/sv_scores.json'))
for k in ['best_score','median','fraction_ge_075','fraction_ge_080']:
    v = [r[k] for r in rows]
    q = statistics.quantiles(v, n=4)
    print(k, 'min', min(v), 'p25', q[0], 'med', statistics.median(v), 'p75', q[2], 'max', max(v))
for th in [.70,.72,.74,.75,.76,.78,.80]:
    print('best>=', th, sum(r['best_score'] >= th for r in rows), 'median>=', sum(r['median'] >= th for r in rows))
print('top')
for r in sorted(rows, key=lambda r:(r['median'],r['best_score']), reverse=True)[:40]:
    print(f"{r['median']:.3f} {r['best_score']:.3f} {r['fraction_ge_075']:.2f} {r['name']}")
