from __future__ import annotations

import numpy as np
import torch
from pathlib import Path


ROOT = Path(r"radio_drama_work/radio_drama_work")
V2 = ROOT / "full_funasr_target_recall_v2"
REF = V2 / "embedding_cache" / "all_target_references.npz"
NEG = ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"


def norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "n": int(values.size),
        "p01": round(float(np.quantile(values, 0.01)), 4),
        "p05": round(float(np.quantile(values, 0.05)), 4),
        "p10": round(float(np.quantile(values, 0.10)), 4),
        "median": round(float(np.median(values)), 4),
        "p90": round(float(np.quantile(values, 0.90)), 4),
        "p95": round(float(np.quantile(values, 0.95)), 4),
        "p99": round(float(np.quantile(values, 0.99)), 4),
    }


def topn(scores: torch.Tensor, n: int) -> torch.Tensor:
    return torch.topk(scores, k=min(n, scores.shape[-1]), dim=-1).values.mean(dim=-1)


def main() -> None:
    refs = norm(torch.from_numpy(np.load(str(REF), allow_pickle=True)["embeddings"]).float())
    centroid = norm(refs.mean(0, keepdim=True))[0]
    pair = refs @ refs.T
    mask = ~torch.eye(len(refs), dtype=torch.bool)
    loo = pair[mask].reshape(len(refs), len(refs) - 1)
    target_top4 = topn(loo, 4).numpy()
    target_top10 = topn(loo, 10).numpy()
    result = {
        "reference_count": len(refs),
        "target_reference_loo_top4": summary(target_top4),
        "target_reference_loo_top10": summary(target_top10),
        "target_reference_centroid": summary((refs @ centroid).numpy()),
    }
    if NEG.exists():
        neg = norm(torch.from_numpy(np.load(str(NEG), allow_pickle=True)["embeddings"]).float())
        scores = neg @ refs.T
        result["hard_negative_to_target_max"] = summary(scores.max(dim=1).values.numpy())
        result["hard_negative_to_target_top4"] = summary(topn(scores, 4).numpy())
        result["hard_negative_to_target_top10"] = summary(topn(scores, 10).numpy())
        result["hard_negative_centroid"] = summary((neg @ centroid).numpy())
    print(result)


if __name__ == "__main__":
    main()
