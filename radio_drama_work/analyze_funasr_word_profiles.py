from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from funasr import AutoModel

import sys
sys.path.insert(0, str(Path(r"radio_drama_work/radio_drama_work")))
import full_speaker_diarize_new_v1 as base


ROOT = Path(r"radio_drama_work/radio_drama_work")
V2 = ROOT / "full_funasr_target_recall_v2"
PKG = Path(r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50")
SOURCE = ROOT / "new_radio_batch_v1" / "vocals" / "1405372904-1-192_vocals.wav"


def main() -> None:
    report = json.loads((V2 / "report.json").read_text(encoding="utf-8"))
    source_report = next(row for row in report["sources"] if row["source"].endswith(SOURCE.name))
    rows = source_report["word_rows"]
    refs = base.norm_rows(torch.from_numpy(np.load(str(V2 / "embedding_cache" / "all_target_references.npz"))["embeddings"]).float())
    centroid = base.norm_rows(refs.mean(0, keepdim=True))[0]
    audio, sr = base.read_mono(SOURCE)
    audio16 = base.resample_1d(audio, sr, base.CAM_SR)
    valid = [row for row in rows if row["end"] - row["start"] >= 0.16]
    waves = [base.segment(audio16, base.CAM_SR, row["start"], row["end"], 1.20) for row in valid]
    cam = AutoModel(model=str(base.CAM_MODEL), device="cuda:0", check_latest=False, disable_pbar=True)
    embs = base.cam_embed_batch(cam, waves)
    waves_short = [base.segment(audio16, base.CAM_SR, row["start"], row["end"], 0.80) for row in valid]
    embs_short = base.cam_embed_batch(cam, waves_short)
    candidate_rows = [row for row in valid if row["decision"] in {"accepted", "expanded"}]
    candidate_waves = []
    candidate_keys = []
    for row in candidate_rows:
        for center in (row["start"], (row["start"] + row["end"]) / 2.0, row["end"]):
            candidate_waves.append(base.segment(audio16, base.CAM_SR, center - 0.4, center + 0.4, 0.8))
            candidate_keys.append((row["start"], center))
    candidate_embs = base.cam_embed_batch(cam, candidate_waves)
    groups: dict[str, list[float]] = defaultdict(list)
    groups_top4: dict[str, list[float]] = defaultdict(list)
    groups_top10: dict[str, list[float]] = defaultdict(list)
    groups_short_top4: dict[str, list[float]] = defaultdict(list)
    groups_short_max: dict[str, list[float]] = defaultdict(list)
    for row, emb in zip(valid, embs):
        groups[row["decision"]].append(float(emb @ centroid))
        sims = torch.sort(emb @ refs.T).values
        groups_top4[row["decision"]].append(float(sims[-4:].mean()))
        groups_top10[row["decision"]].append(float(sims[-10:].mean()))
    for row, emb in zip(valid, embs_short):
        sims = torch.sort(emb @ refs.T).values
        groups_short_top4[row["decision"]].append(float(sims[-4:].mean()))
        groups_short_max[row["decision"]].append(float(sims[-1]))
    for decision, values in sorted(groups.items()):
        a = np.asarray(values)
        print(decision, {"n": int(len(a)), "p10": round(float(np.quantile(a, .10)), 4), "p25": round(float(np.quantile(a, .25)), 4), "median": round(float(np.median(a)), 4), "p75": round(float(np.quantile(a, .75)), 4), "p90": round(float(np.quantile(a, .90)), 4), "ge_060": int(np.sum(a >= .60)), "ge_065": int(np.sum(a >= .65)), "ge_070": int(np.sum(a >= .70))})
        b = np.asarray(groups_top4[decision])
        c = np.asarray(groups_top10[decision])
        print("  top4", {"p10": round(float(np.quantile(b, .10)), 4), "median": round(float(np.median(b)), 4), "p90": round(float(np.quantile(b, .90)), 4), "ge_060": int(np.sum(b >= .60)), "ge_065": int(np.sum(b >= .65)), "ge_070": int(np.sum(b >= .70))})
        print("  top10", {"p10": round(float(np.quantile(c, .10)), 4), "median": round(float(np.median(c)), 4), "p90": round(float(np.quantile(c, .90)), 4), "ge_060": int(np.sum(c >= .60)), "ge_065": int(np.sum(c >= .65)), "ge_070": int(np.sum(c >= .70))})
        d = np.asarray(groups_short_top4[decision])
        e = np.asarray(groups_short_max[decision])
        print("  short_top4", {"p10": round(float(np.quantile(d, .10)), 4), "median": round(float(np.median(d)), 4), "p90": round(float(np.quantile(d, .90)), 4), "ge_050": int(np.sum(d >= .50)), "ge_055": int(np.sum(d >= .55)), "ge_060": int(np.sum(d >= .60))})
        print("  short_max", {"p10": round(float(np.quantile(e, .10)), 4), "median": round(float(np.median(e)), 4), "p90": round(float(np.quantile(e, .90)), 4), "ge_060": int(np.sum(e >= .60)), "ge_065": int(np.sum(e >= .65)), "ge_070": int(np.sum(e >= .70))})
    print("accepted_below_threshold")
    for threshold in (.55, .60, .625, .65, .675, .70):
        selected = [row for row, emb in zip(valid, embs) if row["decision"] == "accepted" and float(emb @ centroid) < threshold]
        print(threshold, len(selected), round(sum(row["end"] - row["start"] for row in selected), 2))
    print("selected_words")
    for row, emb, emb_short in zip(valid, embs, embs_short):
        if row["text"] in {"はい", "飲"} and ((18 <= row["start"] <= 22) or (400 <= row["start"] <= 405)):
            sims = torch.sort(emb @ refs.T).values
            sims_short = torch.sort(emb_short @ refs.T).values
            centers = [row["start"] - 0.2, row["start"], (row["start"] + row["end"]) / 2.0, row["end"], row["end"] + 0.2]
            sweep = base.cam_embed_batch(cam, [base.segment(audio16, base.CAM_SR, center - 0.4, center + 0.4, 0.8) for center in centers])
            sweep_scores = []
            for sweep_emb in sweep:
                sweep_sims = torch.sort(sweep_emb @ refs.T).values
                sweep_scores.append(round(float(sweep_sims[-4:].mean()), 4))
            print({"text": row["text"], "start": row["start"], "end": row["end"], "decision": row["decision"], "long_max": round(float(sims[-1]), 4), "long_top4": round(float(sims[-4:].mean()), 4), "short_max": round(float(sims_short[-1]), 4), "short_top4": round(float(sims_short[-4:].mean()), 4), "sweep_top4": sweep_scores})
    local_rows = []
    for i, row in enumerate(candidate_rows):
        scores = []
        for emb in candidate_embs[i * 3 : i * 3 + 3]:
            values = torch.sort(emb @ refs.T).values
            scores.append(float(values[-4:].mean()))
        local_rows.append((row, scores))
    for decision in ("accepted", "expanded"):
        values = [scores for row, scores in local_rows if row["decision"] == decision]
        print("local_three", decision, {"n": len(values), "center_lt050": sum(scores[1] < .50 for scores in values), "edge_lt048": sum(min(scores[0], scores[2]) < .48 for scores in values), "both_rule": sum(scores[1] < .50 or min(scores[0], scores[2]) < .48 for scores in values), "median_center": round(float(np.median([scores[1] for scores in values])), 4) if values else None})
    directional = 0
    for i, (row, scores) in enumerate(local_rows):
        if row["decision"] != "accepted":
            continue
        next_row = local_rows[i + 1][0] if i + 1 < len(local_rows) else None
        prev_row = local_rows[i - 1][0] if i else None
        right_drop = scores[1] - scores[2] >= .04 and next_row is not None and next_row["start"] - row["end"] <= .25
        left_drop = scores[1] - scores[0] >= .04 and prev_row is not None and row["start"] - prev_row["end"] <= .25
        directional += int(right_drop or left_drop)
    print("directional_boundary_rule", directional)


if __name__ == "__main__":
    main()
