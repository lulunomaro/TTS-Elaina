#!/bin/bash
set -euo pipefail
BASE=/root/autodl-tmp/ireina_new_materials_v1
IN=$BASE/slices
OUT=$BASE/asr_candidates
SV=$BASE/sv_scores.json
mkdir -p "$OUT"
"/root/miniconda3/envs/gptsovits/bin/python" - "$IN" "$OUT" "$SV" <<'PY'
import json, os, sys
inp, out, score_path = sys.argv[1:]
rows = json.load(open(score_path, encoding='utf-8'))
kept = []
for r in rows:
    if r['best_score'] >= 0.70 and (r['median'] >= 0.55 or r['fraction_ge_075'] >= 0.33):
        src = r['file']; dst = os.path.join(out, r['name'])
        if not os.path.exists(dst): os.symlink(src, dst)
        kept.append(r)
print('asr_candidates', len(kept), flush=True)
for r in sorted(kept, key=lambda x:(x['median'], x['best_score']), reverse=True)[:20]:
    print(f"{r['median']:.3f}/t{r['best_score']:.3f}/t{r['name']}", flush=True)
PY
