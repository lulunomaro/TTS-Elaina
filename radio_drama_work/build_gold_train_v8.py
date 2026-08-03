"""Build a conservative, training-only set from the v7 accepted words.

The earlier v7 page intentionally contained three tiers.  This pass starts
from v7's accepted tier only, then re-checks every underlying ASR word with
the 168 positive references and the confirmed hard-negative bank.  Audio is
copied from the original BS-Roformer vocals; no source separation or voice
reconstruction is performed here.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
sys.path.insert(0, str(ROOT))
import full_speaker_diarize_new_v1 as base

V2_REPORT = ROOT / "full_funasr_target_recall_v2" / "report.json"
V7_REPORT = ROOT / "full_funasr_local_boundary_v7" / "report.json"
REF_CACHE = ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / "all_target_references.npz"
NEG_CACHE = ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"
LOCAL_CACHE = ROOT / "full_funasr_local_boundary_v4" / "embedding_cache"
OUT = ROOT / "full_funasr_gold_v9"
ACCEPTED = OUT / "accepted_segments"
REJECTED = OUT / "rejected"
REPORT = OUT / "report.json"
LIST = ROOT / "full_funasr_gold_train_candidate_v9.list"

OUTPUT_SR = 32000
MIN_CLIP_SEC = 1.80
MAX_CLIP_SEC = 10.0
MERGE_GAP_SEC = 0.45
EDGE_PAD_SEC = 0.16
TAIL_PAD_SEC = 0.12
SOURCE_MARGIN_MIN = 0.04

# Calibrated against the target leave-one-out and hard-negative score ranges.
# A word has to be strong in the middle, remain target-like at both edges,
# and agree with the source-local target-vs-other speaker prototypes.
MID_TOP4_MIN = 0.52
MIN_TOP4_MIN = 0.48
HARD_MAX = 0.65
MID_GAP_MIN = -0.01
MIN_GAP_MIN = -0.10
SOURCE_MID_GAP_MIN = 0.05
SOURCE_MIN_GAP_MIN = -0.05


def norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def forbidden(row: dict, start: float, end: float) -> bool:
    return any(overlap(start, end, float(x["start"]), float(x["end"])) for x in row.get("hard_exclusions", []))


def source_margin(source: dict) -> float:
    margins = []
    for run in source.get("cluster_runs", []):
        values = sorted((float(c["target_top4"]) for c in run.get("clusters", [])), reverse=True)
        if len(values) >= 2:
            margins.append(values[0] - values[1])
    return min(margins) if margins else 0.0


def write_clip(audio: np.ndarray, sr: int, start: float, end: float, path: Path) -> None:
    raw = base.segment(audio, sr, start, end)
    out = base.resample_1d(raw, sr, OUTPUT_SR)
    sf.write(str(path), out, OUTPUT_SR, subtype="PCM_16")


def line(path: Path, text: str) -> str:
    return f"{path}|full_funasr_gold_v9|JA|{text.replace(chr(10), ' ').strip()}"


def main() -> None:
    for path in (V2_REPORT, V7_REPORT, REF_CACHE, NEG_CACHE):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (OUT, ACCEPTED, REJECTED):
        path.mkdir(parents=True, exist_ok=True)

    refs = norm(torch.from_numpy(np.load(str(REF_CACHE), allow_pickle=True)["embeddings"]).float())
    neg = norm(torch.from_numpy(np.load(str(NEG_CACHE), allow_pickle=True)["embeddings"]).float())
    v2 = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    v7 = json.loads(V7_REPORT.read_text(encoding="utf-8"))
    v7_intervals = [(str(x["source"]), float(x["start"]), float(x["end"])) for s in v7.get("sources", []) for x in s.get("accepted_segments", [])]

    report = {
        "version": "full_funasr_gold_v9",
        "policy": "v7 accepted words only; 168 positive references; 13 confirmed hard-negative windows; source-local target-vs-other prototypes; local CAM++ start/mid/end score and target-negative margin; guarded sources excluded",
        "input_waveform": "original BS-Roformer vocals",
        "target_reference_count": int(len(refs)),
        "hard_negative_count": int(len(neg)),
        "parameters": {"mid_top4_min": MID_TOP4_MIN, "min_top4_min": MIN_TOP4_MIN, "hard_max": HARD_MAX, "mid_gap_min": MID_GAP_MIN, "min_gap_min": MIN_GAP_MIN, "source_mid_gap_min": SOURCE_MID_GAP_MIN, "source_min_gap_min": SOURCE_MIN_GAP_MIN, "source_margin_min": SOURCE_MARGIN_MIN, "min_clip_sec": MIN_CLIP_SEC, "merge_gap_sec": MERGE_GAP_SEC},
        "sources": [],
    }
    list_lines: list[str] = []
    clip_index = 0

    for s in v2.get("sources", []):
        source = Path(s["source"])
        source_key = str(source)
        accepted_intervals = [(st, en) for src, st, en in v7_intervals if src == source_key]
        margin = source_margin(s)
        guarded = margin < SOURCE_MARGIN_MIN or not accepted_intervals
        audio, sr = base.read_mono(source)
        duration = len(audio) / sr
        cache = LOCAL_CACHE / f"{source.stem}.npz"
        local = norm(torch.from_numpy(np.load(str(cache), allow_pickle=True)["embeddings"]).float()).reshape(-1, 3, refs.shape[1])
        v2_cache = np.load(str(ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / f"{source.stem}.npz"), allow_pickle=True)
        chunk_embs = norm(torch.from_numpy(v2_cache["embeddings"]).float())
        target_idx = [int(x["index"]) for x in s.get("chunk_rows", []) if x.get("tier") == "target"]
        other_idx = [int(x["index"]) for x in s.get("chunk_rows", []) if x.get("tier") == "other"]
        target_center = norm(chunk_embs[target_idx].mean(0, keepdim=True))[0] if target_idx else torch.zeros(refs.shape[1])
        other_center = norm(chunk_embs[other_idx].mean(0, keepdim=True))[0] if other_idx else torch.zeros(refs.shape[1])
        words = [r for r in sorted(s.get("word_rows", []), key=lambda r: (float(r["start"]), float(r["end"]))) if r.get("decision") in {"accepted", "expanded"}]
        eligible = [any(st <= float(r["start"]) + 0.01 and en >= float(r["end"]) - 0.01 for st, en in accepted_intervals) for r in words]
        passed: list[dict] = []
        reasons: Counter[str] = Counter()
        for i, row in enumerate(words):
            item = dict(row)
            item["source"] = source_key
            if guarded:
                item["gold_pass"] = False
                item["gold_reason"] = "guarded_source_or_no_v7_acceptance"
                reasons[item["gold_reason"]] += 1
                passed.append(item)
                continue
            if not eligible[i]:
                item["gold_pass"] = False
                item["gold_reason"] = "not_in_v7_accepted_segment"
                reasons[item["gold_reason"]] += 1
                passed.append(item)
                continue
            scores = torch.sort(local[i] @ refs.T, dim=1).values[:, -4:].mean(1)
            hard = (local[i] @ neg.T).max(1).values
            source_target = local[i] @ target_center
            source_other = local[i] @ other_center
            item.update({"gold_start_top4": float(scores[0]), "gold_mid_top4": float(scores[1]), "gold_end_top4": float(scores[2]), "gold_min_top4": float(scores.min()), "gold_start_hard_max": float(hard[0]), "gold_mid_hard_max": float(hard[1]), "gold_end_hard_max": float(hard[2]), "gold_hard_max": float(hard.max()), "gold_mid_gap": float(scores[1] - hard[1]), "gold_min_gap": float(scores.min() - hard.max()), "source_mid_target": float(source_target[1]), "source_min_target": float(source_target.min()), "source_mid_other": float(source_other[1]), "source_max_other": float(source_other.max()), "source_mid_gap": float(source_target[1] - source_other[1]), "source_min_gap": float(source_target.min() - source_other.max())})
            failed = []
            if float(scores[1]) < MID_TOP4_MIN: failed.append("mid_target_score")
            if float(scores.min()) < MIN_TOP4_MIN: failed.append("edge_target_score")
            if float(hard.max()) >= HARD_MAX: failed.append("hard_negative_similarity")
            if float(scores[1] - hard[1]) < MID_GAP_MIN: failed.append("mid_target_negative_gap")
            if float(scores.min() - hard.max()) < MIN_GAP_MIN: failed.append("edge_target_negative_gap")
            if float(source_target[1] - source_other[1]) < SOURCE_MID_GAP_MIN: failed.append("source_speaker_gap")
            if float(source_target.min() - source_other.max()) < SOURCE_MIN_GAP_MIN: failed.append("source_edge_speaker_gap")
            item["gold_pass"] = not failed
            item["gold_reason"] = "passed" if not failed else "+".join(failed)
            reasons[item["gold_reason"]] += 1
            passed.append(item)

        source_segments = []
        current: list[dict] = []
        for i, row in enumerate(passed):
            if not row["gold_pass"]:
                if current:
                    source_segments.append(current)
                    current = []
                continue
            if current:
                prev = current[-1]
                if i != int(prev["_idx"]) + 1 or float(row["start"]) - float(prev["end"]) > MERGE_GAP_SEC:
                    source_segments.append(current)
                    current = []
            row["_idx"] = i
            current.append(row)
        if current:
            source_segments.append(current)

        outputs = []
        for group in source_segments:
            first = group[0]
            last = group[-1]
            start = max(0.0, float(first["start"]) - EDGE_PAD_SEC)
            end = min(duration, float(last["end"]) + TAIL_PAD_SEC)
            previous = next((w for w in reversed(s.get("word_rows", [])) if float(w["end"]) <= float(first["start"]) + 1e-5), None)
            following = next((w for w in s.get("word_rows", []) if float(w["start"]) >= float(last["end"]) - 1e-5), None)
            if previous is not None and float(previous["end"]) >= float(first["start"]) - 0.02:
                start = float(first["start"])
            if following is not None and following is not last and float(following["start"]) <= end:
                end = min(end, max(float(last["end"]), float(following["start"]) - 0.02))
            if end - start < MIN_CLIP_SEC or end - start > MAX_CLIP_SEC:
                continue
            if forbidden(s, start, end):
                continue
            name = f"gold_{clip_index:05d}_{source.stem}_{int(round(start * OUTPUT_SR)):010d}_{int(round(end * OUTPUT_SR)):010d}.wav"
            path = ACCEPTED / name
            write_clip(audio, sr, start, end, path)
            text = "".join(str(w.get("text", "")) for w in group)
            outputs.append({"file": str(path), "source": source_key, "start": start, "end": end, "duration": end - start, "text": text, "word_count": len(group), "min_word_mid_top4": min(float(w["gold_mid_top4"]) for w in group), "min_word_target_negative_gap": min(float(w["gold_min_gap"]) for w in group), "tier": "gold_train_candidate_v9"})
            list_lines.append(line(path, text))
            clip_index += 1
        report["sources"].append({"source": source_key, "duration_sec": duration, "source_margin_min": margin, "guarded_from_training": guarded, "v7_accepted_interval_count": len(accepted_intervals), "eligible_word_count": sum(eligible), "gold_pass_word_count": sum(bool(x["gold_pass"]) for x in passed), "gold_segment_count": len(outputs), "gold_duration_sec": sum(float(x["duration"]) for x in outputs), "failure_reasons": dict(reasons), "segments": outputs})
        print(f"[source] {source.name} guarded={guarded} eligible={sum(eligible)} words_pass={sum(bool(x['gold_pass']) for x in passed)} segments={len(outputs)} sec={sum(float(x['duration']) for x in outputs):.1f}", flush=True)

    LIST.write_text("/n".join(list_lines) + ("/n" if list_lines else ""), encoding="utf-8")
    report["candidate_list"] = str(LIST)
    report["candidate_count"] = len(list_lines)
    report["candidate_total_sec"] = sum(sf.info(str(Path(line.split("|", 1)[0]))).duration for line in list_lines)
    report["validation"] = {"missing": [line.split("|", 1)[0] for line in list_lines if not Path(line.split("|", 1)[0]).exists()], "duplicates": [p for p, n in Counter(line.split("|", 1)[0] for line in list_lines).items() if n > 1]}
    report["validation"]["passed"] = not report["validation"]["missing"] and not report["validation"]["duplicates"]
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": report["candidate_count"], "candidate_total_sec": report["candidate_total_sec"], "candidate_list": str(LIST), "report": str(REPORT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
