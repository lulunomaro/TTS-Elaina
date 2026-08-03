from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from funasr.models.campplus.cluster_backend import ClusterBackend

ROOT = Path(r"radio_drama_work/radio_drama_work")
V2 = ROOT / "full_funasr_target_recall_v2"
REF_CACHE = V2 / "embedding_cache" / "all_target_references.npz"


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    refs = norm_rows(torch.from_numpy(np.load(str(REF_CACHE), allow_pickle=True)["embeddings"]).float())
    centroid = norm_rows(refs.mean(0, keepdim=True))[0]
    for name in ("1405372904-1-192_vocals",):
        cached = np.load(str(V2 / "embedding_cache" / f"{name}.npz"))
        embs = norm_rows(torch.from_numpy(cached["embeddings"]).float())
        print(f"SOURCE {name} chunks={len(embs)}")
        for k in (3, 4, 5, 6, 7, 8, 10):
            labels = np.asarray(ClusterBackend()(embs.numpy(), oracle_num=k), dtype=int)
            print(f"  K={k}")
            for label in sorted(set(labels.tolist())):
                center = norm_rows(embs[labels == label].mean(0, keepdim=True))[0]
                sims = (center @ refs.T).numpy()
                print({
                    "cluster": int(label),
                    "count": int(np.sum(labels == label)),
                    "centroid": round(float(center @ centroid), 4),
                    "max": round(float(np.max(sims)), 4),
                    "top4": round(float(np.mean(np.sort(sims)[-4:])), 4),
                    "top10": round(float(np.mean(np.sort(sims)[-10:])), 4),
                    "q90": round(float(np.quantile(sims, 0.90)), 4),
                    "median": round(float(np.median(sims)), 4),
                })
            best = max(set(labels.tolist()), key=lambda label: float(norm_rows(embs[labels == label].mean(0, keepdim=True))[0] @ centroid))
            values = (embs[labels == best] @ centroid).numpy()
            print("  best_chunk_centroid", {"cluster": int(best), "p10": round(float(np.quantile(values, .10)), 4), "p25": round(float(np.quantile(values, .25)), 4), "median": round(float(np.median(values)), 4), "p75": round(float(np.quantile(values, .75)), 4), "p90": round(float(np.quantile(values, .90)), 4), "ge_070": int(np.sum(values >= .70)), "ge_075": int(np.sum(values >= .75)), "ge_080": int(np.sum(values >= .80))})


if __name__ == "__main__":
    main()
