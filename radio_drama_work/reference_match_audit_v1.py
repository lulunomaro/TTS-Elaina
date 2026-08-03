# -*- coding: utf-8 -*-
"""Compare new candidates with the old target-speaker material and refs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"radio_drama_work/radio_drama_work")
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
LIST = ROOT / "new_radio_quality_train_candidate_v1.list"
OLD_DIR = ROOT / "dataset_v1" / "wavs"
OUT = ROOT / "new_radio_quality_v1" / "reference_match_audit_v1.json"


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {key: round(float(value), 4) for key, value in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), np.quantile(array, [0, .1, .25, .5, .75, .9, 1]))}


def score_embedding(embedding: torch.Tensor, refs: torch.Tensor, negatives: torch.Tensor, ref_center: torch.Tensor, old_center: torch.Tensor) -> dict:
    similarities = embedding @ refs.T
    center = float((embedding @ ref_center).item())
    p25 = float(torch.quantile(similarities, .25).item())
    negative_max = float((embedding @ negatives.T).max().item()) if len(negatives) else -1.0
    old_similarity = float((embedding @ old_center).item())
    return {
        "center": center,
        "p25": p25,
        "negative_max": negative_max,
        "margin": center - negative_max,
        "old_centroid_similarity": old_similarity,
    }


def main() -> None:
    os.chdir(PKG)
    sys.path.insert(0, str(ROOT))
    import word_target_hardcut as sv  # noqa: E402

    refs_paths = sv.load_refs()
    negative_paths = sv.load_negatives()
    refs = torch.cat([sv.embed_file(path) for path in refs_paths], dim=0)
    negatives = torch.cat([sv.embed_file(path) for path in negative_paths], dim=0) if negative_paths else torch.zeros((1, refs.shape[1]))
    ref_center = torch.nn.functional.normalize(refs.mean(0), dim=-1)

    old_paths = sorted(OLD_DIR.glob("old_my_voice*.wav"))
    old_embeddings = []
    old_rows = []
    for index, path in enumerate(old_paths, 1):
        embedding = sv.embed_file(path)
        old_embeddings.append(embedding)
        old_rows.append({"file": str(path), **score_embedding(embedding, refs, negatives, ref_center, torch.zeros(refs.shape[1]))})
        if index % 25 == 0 or index == len(old_paths):
            print(f"old {index}/{len(old_paths)}", flush=True)
    old_center = torch.nn.functional.normalize(torch.cat(old_embeddings, dim=0).mean(0), dim=-1)
    for row, embedding in zip(old_rows, old_embeddings):
        row["old_centroid_similarity"] = float((embedding @ old_center).item())

    candidate_paths = [Path(line.split("|", 1)[0].strip()) for line in LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidate_rows = []
    for index, path in enumerate(candidate_paths, 1):
        embedding = sv.embed_file(path)
        candidate_rows.append({"file": str(path), **score_embedding(embedding, refs, negatives, ref_center, old_center)})
        if index % 25 == 0 or index == len(candidate_paths):
            print(f"candidate {index}/{len(candidate_paths)}", flush=True)

    report = {
        "version": "reference_match_audit_v1",
        "reference_count": len(refs_paths),
        "negative_count": len(negative_paths),
        "old_material_count": len(old_rows),
        "candidate_count": len(candidate_rows),
        "old_material_distribution": {key: quantiles([row[key] for row in old_rows]) for key in ("center", "p25", "margin", "old_centroid_similarity")},
        "candidate_distribution": {key: quantiles([row[key] for row in candidate_rows]) for key in ("center", "p25", "margin", "old_centroid_similarity")},
        "candidate_strict_pass_count": sum(row["center"] >= .72 and row["p25"] >= .50 and row["margin"] >= .02 for row in candidate_rows),
        "candidate_rows": candidate_rows,
        "old_rows": old_rows,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OLD_DISTRIBUTION", report["old_material_distribution"], flush=True)
    print("CANDIDATE_DISTRIBUTION", report["candidate_distribution"], flush=True)
    print("CANDIDATE_STRICT_PASS", report["candidate_strict_pass_count"], flush=True)
    for row in sorted(candidate_rows, key=lambda item: item["old_centroid_similarity"])[:10]:
        print("LOW", Path(row["file"]).name, {key: round(row[key], 3) for key in ("center", "p25", "margin", "old_centroid_similarity")}, flush=True)
    for row in sorted(candidate_rows, key=lambda item: item["old_centroid_similarity"], reverse=True)[:10]:
        print("HIGH", Path(row["file"]).name, {key: round(row[key], 3) for key in ("center", "p25", "margin", "old_centroid_similarity")}, flush=True)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
