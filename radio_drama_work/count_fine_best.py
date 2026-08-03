import json
from collections import Counter
rows = json.load(open(r"radio_drama_work/new_candidates_fine_sv.json", encoding="utf-8"))
for th in (.72,.74,.75,.76,.78,.80):
    keep=[r for r in rows if r['best_score']>=th]
    print(th,len(keep),Counter(r['name'].split('_vocals')[0] for r in keep))
