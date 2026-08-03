"""Recover guarded-source clips with stricter word-level checks.

The two sources were not discarded wholesale because they contain no target
voice; they were guarded because the source-level cluster margin was weak.
This pass therefore requires stronger local target-vs-negative and
target-vs-other-speaker margins for every retained word, then merges only
adjacent passing words. It uses the original BS-Roformer vocals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(r"radio_drama_work/radio_drama_work")
sys.path.insert(0, str(ROOT))
import full_speaker_diarize_new_v1 as base

V2 = ROOT / "full_funasr_target_recall_v2" / "report.json"
REF = ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / "all_target_references.npz"
NEG = ROOT / "full_speaker_diarize_new_v3" / "embedding_cache" / "hard_negatives.npz"
LOCAL = ROOT / "full_funasr_local_boundary_v4" / "embedding_cache"
OUT = ROOT / "full_funasr_guarded_strict_v1"
ACCEPTED = OUT / "accepted_segments"
LIST = ROOT / "full_funasr_guarded_strict_train_candidate_v1.list"
REPORT = OUT / "report.json"

OUTPUT_SR = 32000
MIN_CLIP_SEC = 1.80
MERGE_GAP_SEC = 0.45
EDGE_PAD_SEC = 0.16
TAIL_PAD_SEC = 0.12
MIN_WORDS = 2

RULE = {"mid_top4_min": .54, "min_top4_min": .50, "hard_max": .62, "mid_gap_min": .01, "min_gap_min": -.04, "source_mid_gap_min": .08, "source_min_gap_min": -.02}


def norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def overlap(a: float, b: float, c: float, d: float) -> bool:
    return max(a, c) < min(b, d)


def forbidden(source: dict, start: float, end: float) -> bool:
    return any(overlap(start, end, float(x["start"]), float(x["end"])) for x in source.get("hard_exclusions", []))


def write_clip(audio: np.ndarray, sr: int, start: float, end: float, path: Path) -> None:
    sf.write(str(path), base.resample_1d(base.segment(audio, sr, start, end), sr, OUTPUT_SR), OUTPUT_SR, subtype="PCM_16")


def main() -> None:
    for path in (V2, REF, NEG):
        if not path.exists():
            raise FileNotFoundError(path)
    ACCEPTED.mkdir(parents=True, exist_ok=True)
    refs = norm(torch.from_numpy(np.load(str(REF), allow_pickle=True)["embeddings"]).float())
    neg = norm(torch.from_numpy(np.load(str(NEG), allow_pickle=True)["embeddings"]).float())
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    report = {"version": "full_funasr_guarded_strict_v1", "policy": "FunASR word timestamps + CAM++ local start/mid/end + 168 target refs + hard negatives + source-local other-speaker prototype; raw BS-Roformer waveform; no ClearVoice reconstruction", "parameters": {**RULE, "min_clip_sec": MIN_CLIP_SEC, "merge_gap_sec": MERGE_GAP_SEC, "min_words": MIN_WORDS}, "sources": []}
    lines = []
    index = 0
    for s in v2["sources"]:
        source = Path(s["source"])
        if source.name not in {"1405372904-1-192_vocals.wav", "306672160-1-192_vocals.wav"}:
            continue
        audio, sr = base.read_mono(source)
        duration = len(audio) / sr
        local = norm(torch.from_numpy(np.load(str(LOCAL / f"{source.stem}.npz"), allow_pickle=True)["embeddings"]).float()).reshape(-1, 3, refs.shape[1])
        chunk_cache = np.load(str(ROOT / "full_funasr_target_recall_v2" / "embedding_cache" / f"{source.stem}.npz"), allow_pickle=True)
        chunks = norm(torch.from_numpy(chunk_cache["embeddings"]).float())
        ti = [int(x["index"]) for x in s["chunk_rows"] if x.get("tier") == "target"]
        oi = [int(x["index"]) for x in s["chunk_rows"] if x.get("tier") == "other"]
        target_center = norm(chunks[ti].mean(0, keepdim=True))[0]
        other_center = norm(chunks[oi].mean(0, keepdim=True))[0]
        words = [r for r in sorted(s["word_rows"], key=lambda r: (float(r["start"]), float(r["end"]))) if r.get("decision") in {"accepted", "expanded"}]
        selected = []
        counts = {"candidate_words": len(words), "passed_words": 0, "failed_words": 0, "segments": 0}
        for i, word in enumerate(words):
            e = local[i]
            top4 = torch.sort(e @ refs.T, dim=1).values[:, -4:].mean(1)
            hard = (e @ neg.T).max(1).values
            st = e @ target_center
            so = e @ other_center
            metrics = {"mid": float(top4[1]), "min": float(top4.min()), "hard": float(hard.max()), "mid_gap": float(top4[1] - hard[1]), "min_gap": float(top4.min() - hard.max()), "src_mid_gap": float(st[1] - so[1]), "src_min_gap": float(st.min() - so.max())}
            ok = metrics["mid"] >= RULE["mid_top4_min"] and metrics["min"] >= RULE["min_top4_min"] and metrics["hard"] < RULE["hard_max"] and metrics["mid_gap"] >= RULE["mid_gap_min"] and metrics["min_gap"] >= RULE["min_gap_min"] and metrics["src_mid_gap"] >= RULE["source_mid_gap_min"] and metrics["src_min_gap"] >= RULE["source_min_gap_min"]
            if ok:
                selected.append((i, word, metrics))
                counts["passed_words"] += 1
            else:
                counts["failed_words"] += 1
        groups = []
        current = []
        for item in selected:
            if current and (item[0] != current[-1][0] + 1 or float(item[1]["start"]) - float(current[-1][1]["end"]) > MERGE_GAP_SEC):
                groups.append(current)
                current = []
            current.append(item)
        if current:
            groups.append(current)
        outputs = []
        for group in groups:
            if len(group) < MIN_WORDS:
                continue
            first, last = group[0][1], group[-1][1]
            start = max(0.0, float(first["start"]) - EDGE_PAD_SEC)
            end = min(duration, float(last["end"]) + TAIL_PAD_SEC)
            previous = next((w for w in reversed(s["word_rows"]) if float(w["end"]) <= float(first["start"]) + 1e-5), None)
            following = next((w for w in s["word_rows"] if float(w["start"]) >= float(last["end"]) - 1e-5), None)
            if previous is not None and float(previous["end"]) >= float(first["start"]) - 0.02:
                start = float(first["start"])
            if following is not None and float(following["start"]) <= end:
                end = min(end, max(float(last["end"]), float(following["start"]) - 0.02))
            if end - start < MIN_CLIP_SEC or forbidden(s, start, end):
                continue
            path = ACCEPTED / f"guarded_{index:05d}_{source.stem}_{int(round(start * OUTPUT_SR)):010d}_{int(round(end * OUTPUT_SR)):010d}.wav"
            write_clip(audio, sr, start, end, path)
            text = "".join(str(x[1].get("text", "")) for x in group)
            outputs.append({"file": str(path), "source": str(source), "start": start, "end": end, "duration": end - start, "text": text, "word_count": len(group), "min_mid_top4": min(x[2]["mid"] for x in group), "min_source_mid_gap": min(x[2]["src_mid_gap"] for x in group), "tier": "guarded_strict_auto_candidate_v1"})
            lines.append(f"{path}|full_funasr_guarded_strict_v1|JA|{text}")
            index += 1
        counts["segments"] = len(outputs)
        report["sources"].append({"source": str(source), "duration_sec": duration, "guarded": True, **counts, "accepted_segments": outputs})
        print(source.name, counts, "sec", round(sum(x["duration"] for x in outputs), 2), flush=True)
    LIST.write_text("/n".join(lines) + ("/n" if lines else ""), encoding="utf-8")
    report.update({"candidate_list": str(LIST), "candidate_count": len(lines), "candidate_total_sec": sum(float(x["duration"]) for s in report["sources"] for x in s["accepted_segments"]), "validation": {"missing": [x.split("|", 1)[0] for x in lines if not Path(x.split("|", 1)[0]).exists()]}})
    report["validation"]["passed"] = not report["validation"]["missing"]
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": report["candidate_count"], "candidate_total_sec": report["candidate_total_sec"], "list": str(LIST), "report": str(REPORT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
