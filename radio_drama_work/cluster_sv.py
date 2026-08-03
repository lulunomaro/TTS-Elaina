import glob, os, torch
import torch.nn.functional as F
ps = glob.glob('/root/autodl-tmp/exp/ireina_v1/7-sv_cn/*.pt')
d = {}
for p in ps:
    base = os.path.basename(p)
    k = 'radio' if 'radio_' in base else 'old'
    x = torch.load(p, map_location='cpu', weights_only=False).float().flatten()
    d.setdefault(k, []).append(F.normalize(x, dim=0))
for k, v in d.items():
    print(k, len(v), tuple(v[0].shape))
for a in d:
    for b in d:
        z, w = torch.stack(d[a]), torch.stack(d[b])
        print(a, b, 'mean_cos', float((z @ w.T).mean()), 'centroid_cos', float(F.normalize(z.mean(0), dim=0) @ F.normalize(w.mean(0), dim=0)))
# within-group spread and nearest cross-group similarity
for k, v in d.items():
    z = torch.stack(v); sim = z @ z.T
    mask = ~torch.eye(len(z), dtype=torch.bool)
    print(k, 'within_mean', float(sim[mask].mean()), 'within_min', float(sim[mask].min()), 'within_max', float(sim[mask].max()))
if 'old' in d and 'radio' in d:
    z, w = torch.stack(d['old']), torch.stack(d['radio'])
    print('cross_nearest_old_mean', float((z @ w.T).max(1).values.mean()))
    print('cross_nearest_radio_mean', float((z @ w.T).max(0).values.mean()))
