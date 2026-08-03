from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"radio_drama_work/radio_drama_work")
V2 = ROOT / "full_funasr_target_recall_v2" / "report.json"
V7 = ROOT / "full_funasr_local_boundary_v7" / "report.json"
REF = ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / "all_target_references.npz"
NEG = ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"
LOCAL = ROOT / "full_funasr_local_boundary_v4" / "embedding_cache"


def norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def q(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float32)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "p01": float(np.quantile(a, .01)), "p05": float(np.quantile(a, .05)), "p10": float(np.quantile(a, .10)), "median": float(np.median(a)), "p90": float(np.quantile(a, .90)), "p95": float(np.quantile(a, .95))}


def main() -> None:
    refs = norm(torch.from_numpy(np.load(str(REF), allow_pickle=True)["embeddings"]).float())
    neg = norm(torch.from_numpy(np.load(str(NEG), allow_pickle=True)["embeddings"]).float())
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    v7 = json.loads(V7.read_text(encoding="utf-8"))
    diarized_neg = []
    source_proto = {}
    for s in v2["sources"]:
        data = np.load(str(LOCAL.parent.parent / "full_funasr_target_recall_v2" / "embedding_cache" / (Path(s["source"]).stem + ".npz")), allow_pickle=True)
        embs = norm(torch.from_numpy(data["embeddings"]).float())
        other_idx = [int(x["index"]) for x in s["chunk_rows"] if x.get("tier") == "other"]
        target_idx = [int(x["index"]) for x in s["chunk_rows"] if x.get("tier") == "target"]
        target_center = norm(embs[target_idx].mean(0, keepdim=True))[0] if target_idx else torch.zeros(embs.shape[1])
        other_center = norm(embs[other_idx].mean(0, keepdim=True))[0] if other_idx else torch.zeros(embs.shape[1])
        source_proto[str(s["source"])] = (target_center, other_center)
        diarized_neg.append(embs[other_idx])
    diarized_neg = torch.cat(diarized_neg, dim=0)
    neg_all = torch.cat([neg, diarized_neg], dim=0)
    print("diarized_negative_count", len(diarized_neg))
    accepted = [(str(x["source"]), float(x["start"]), float(x["end"])) for s in v7["sources"] for x in s["accepted_segments"]]
    rows = []
    for s in v2["sources"]:
        source = str(s["source"])
        cache = LOCAL / (Path(source).stem + ".npz")
        data = np.load(str(cache), allow_pickle=True)
        emb = norm(torch.from_numpy(data["embeddings"]).float()).reshape(-1, 3, refs.shape[1])
        candidates = [r for r in sorted(s["word_rows"], key=lambda x: (float(x["start"]), float(x["end"]))) if r.get("decision") in {"accepted", "expanded"}]
        for i, r in enumerate(candidates):
            local_scores = torch.sort(emb[i] @ refs.T, dim=1).values[:, -4:].mean(1)
            hard_scores = (emb[i] @ neg.T).max(1).values
            diarized_scores = (emb[i] @ diarized_neg.T).max(1).values
            all_hard_scores = (emb[i] @ neg_all.T).max(1).values
            target_center, other_center = source_proto[source]
            source_target_scores = emb[i] @ target_center
            source_other_scores = emb[i] @ other_center
            rows.append({
                "source": Path(source).name,
                "start": float(r["start"]), "end": float(r["end"]),
                "accepted_word": any(src == source and st <= float(r["start"]) + 0.01 and en >= float(r["end"]) - 0.01 for src, st, en in accepted),
                "decision": r.get("decision"), "text": r.get("text", ""),
                "ref_long": float(r.get("reference_sim", 0.0)),
                "hard_long": float(r.get("hard_negative_sim", -1.0)),
                "target_vote": float(r.get("target_vote", 0.0)),
                "mid_top4": float(local_scores[1]), "min_top4": float(local_scores.min()), "start_top4": float(local_scores[0]), "end_top4": float(local_scores[2]),
                "mid_hard_max": float(hard_scores[1]), "min_hard_max": float(hard_scores.max()),
                "mid_diarized_hard_max": float(diarized_scores[1]), "min_diarized_hard_max": float(diarized_scores.max()),
                "mid_all_hard_max": float(all_hard_scores[1]), "min_all_hard_max": float(all_hard_scores.max()),
                "src_mid_target": float(source_target_scores[1]), "src_min_target": float(source_target_scores.min()), "src_mid_other": float(source_other_scores[1]), "src_max_other": float(source_other_scores.max()),
            })
            rows[-1]["mid_gap"] = rows[-1]["mid_top4"] - rows[-1]["mid_hard_max"]
            rows[-1]["min_gap"] = rows[-1]["min_top4"] - rows[-1]["min_hard_max"]
            rows[-1]["mid_all_gap"] = rows[-1]["mid_top4"] - rows[-1]["mid_all_hard_max"]
            rows[-1]["min_all_gap"] = rows[-1]["min_top4"] - rows[-1]["min_all_hard_max"]
            rows[-1]["src_mid_gap"] = rows[-1]["src_mid_target"] - rows[-1]["src_mid_other"]
            rows[-1]["src_min_gap"] = rows[-1]["src_min_target"] - rows[-1]["src_max_other"]
    for key in ("mid_top4", "min_top4", "ref_long", "hard_long", "mid_hard_max", "min_hard_max", "mid_gap", "min_gap", "target_vote"):
        print(key, "all", q([float(r[key]) for r in rows]))
        print(key, "accepted", q([float(r[key]) for r in rows if r["accepted_word"]]))
    for key in ("mid_diarized_hard_max", "min_diarized_hard_max", "mid_all_hard_max", "min_all_hard_max", "mid_all_gap", "min_all_gap"):
        print(key, "accepted", q([float(r[key]) for r in rows if r["accepted_word"]]))
    for key in ("src_mid_target", "src_min_target", "src_mid_other", "src_max_other", "src_mid_gap", "src_min_gap"):
        print(key, "accepted", q([float(r[key]) for r in rows if r["accepted_word"]]))
    for threshold in ((.52, .48, .62, .00), (.54, .50, .60, .01), (.56, .52, .60, .02), (.58, .54, .58, .03)):
        mid, mn, hard, gap = threshold
        keep = [r for r in rows if r["accepted_word"] and r["mid_top4"] >= mid and r["min_top4"] >= mn and r["mid_hard_max"] < hard and r["min_hard_max"] < hard and r["mid_gap"] >= gap and r["min_gap"] >= gap]
        print("rule", threshold, "words", len(keep), "sec", sum(float(r["end"] - r["start"]) for r in keep), "sources", sorted({r["source"] for r in keep}))
    for threshold in ((.50, .46, .04, -.06), (.52, .48, .05, -.05), (.54, .50, .06, -.04), (.56, .52, .08, -.03)):
        mid, mn, gap, edge_gap = threshold
        keep = [r for r in rows if r["accepted_word"] and r["mid_top4"] >= mid and r["min_top4"] >= mn and r["src_mid_gap"] >= gap and r["src_min_gap"] >= edge_gap]
        print("source_rule", threshold, "words", len(keep), "sec", sum(float(r["end"] - r["start"]) for r in keep), "sources", sorted({r["source"] for r in keep}))
    for threshold in ((.50, .46, .04, -.06, .65, -.02), (.52, .48, .05, -.05, .65, -.01), (.54, .50, .06, -.04, .65, .00), (.56, .52, .08, -.03, .62, .01)):
        mid, mn, gap, edge_gap, hard, hard_gap = threshold
        keep = [r for r in rows if r["accepted_word"] and r["mid_top4"] >= mid and r["min_top4"] >= mn and r["src_mid_gap"] >= gap and r["src_min_gap"] >= edge_gap and r["min_hard_max"] < hard and r["mid_gap"] >= hard_gap]
        print("combined_rule", threshold, "words", len(keep), "sec", sum(float(r["end"] - r["start"]) for r in keep), "sources", sorted({r["source"] for r in keep}))
    keep = [r for r in rows if r["accepted_word"] and r["mid_top4"] >= .52 and r["src_mid_gap"] >= .05 and r["src_min_gap"] >= -.05 and r["mid_hard_max"] < .65 and r["mid_gap"] >= -.01]
    print("core_only_rule", "words", len(keep), "sec", sum(float(r["end"] - r["start"]) for r in keep), "sources", sorted({r["source"] for r in keep}))
    print("accepted words", sum(r["accepted_word"] for r in rows), "all candidate words", len(rows))


if __name__ == "__main__":
    main()
