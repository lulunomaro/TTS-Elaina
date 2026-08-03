"""Measure conservative word rules on sources v7 had to guard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
sys.path.insert(0, str(ROOT))
import full_speaker_diarize_new_v1 as base

V2 = ROOT / "full_funasr_target_recall_v2" / "report.json"
REF = ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / "all_target_references.npz"
NEG = ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"
LOCAL = ROOT / "full_funasr_local_boundary_v4" / "embedding_cache"


def norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    refs = norm(torch.from_numpy(np.load(str(REF), allow_pickle=True)["embeddings"]).float())
    neg = norm(torch.from_numpy(np.load(str(NEG), allow_pickle=True)["embeddings"]).float())
    report = json.loads(V2.read_text(encoding="utf-8"))
    rows = []
    for source_report in report["sources"]:
        source = Path(source_report["source"])
        if source.name not in {"1405372904-1-192_vocals.wav", "306672160-1-192_vocals.wav"}:
            continue
        local = norm(torch.from_numpy(np.load(str(LOCAL / f"{source.stem}.npz"), allow_pickle=True)["embeddings"]).float()).reshape(-1, 3, refs.shape[1])
        v2_cache = np.load(str(ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / f"{source.stem}.npz"), allow_pickle=True)
        chunks = norm(torch.from_numpy(v2_cache["embeddings"]).float())
        target_idx = [int(x["index"]) for x in source_report["chunk_rows"] if x.get("tier") == "target"]
        other_idx = [int(x["index"]) for x in source_report["chunk_rows"] if x.get("tier") == "other"]
        target_center = norm(chunks[target_idx].mean(0, keepdim=True))[0] if target_idx else torch.zeros(refs.shape[1])
        other_center = norm(chunks[other_idx].mean(0, keepdim=True))[0] if other_idx else torch.zeros(refs.shape[1])
        candidates = [r for r in sorted(source_report["word_rows"], key=lambda r: (float(r["start"]), float(r["end"]))) if r.get("decision") in {"accepted", "expanded"}]
        for i, word in enumerate(candidates):
            e = local[i]
            top4 = torch.sort(e @ refs.T, dim=1).values[:, -4:].mean(1)
            hard = (e @ neg.T).max(1).values
            st = e @ target_center
            so = e @ other_center
            rows.append({"source": source.name, "start": float(word["start"]), "end": float(word["end"]), "text": word.get("text", ""), "mid": float(top4[1]), "min": float(top4.min()), "hard": float(hard.max()), "mid_gap": float(top4[1] - hard[1]), "min_gap": float(top4.min() - hard.max()), "src_mid_gap": float(st[1] - so[1]), "src_min_gap": float(st.min() - so.max())})
    print(f"candidate_words={len(rows)}")
    rules = [
        ("base", .52, .48, .65, -.01, -.10, .05, -.05),
        ("guarded_strict", .54, .50, .62, .01, -.04, .08, -.02),
        ("guarded_very_strict", .56, .52, .60, .02, -.02, .10, 0.00),
        ("identity_first", .50, .46, .65, 0.00, -.08, .06, -.05),
    ]
    for name, mid, mn, hard, gap, edge_gap, src_gap, src_edge_gap in rules:
        keep = [r for r in rows if r["mid"] >= mid and r["min"] >= mn and r["hard"] < hard and r["mid_gap"] >= gap and r["min_gap"] >= edge_gap and r["src_mid_gap"] >= src_gap and r["src_min_gap"] >= src_edge_gap]
        by_source = {s: (sum(r["end"] - r["start"] for r in keep if r["source"] == s), sum(1 for r in keep if r["source"] == s)) for s in sorted({r["source"] for r in rows})}
        print(name, "words", len(keep), "word_sec", round(sum(r["end"] - r["start"] for r in keep), 2), by_source)
    for r in sorted(rows, key=lambda x: x["mid"], reverse=True)[:20]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
