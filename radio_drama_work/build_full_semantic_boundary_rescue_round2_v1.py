# -*- coding: utf-8 -*-
"""Build the short final review for one boundary and two duration/quality rescues."""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import soundfile as sf

import build_full_semantic_acceptance_boundary_rescue_v1 as base
import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


PARENT_AUDIT = ROOT / "full_semantic_acceptance_boundary_rescue_v1" / "audit_manifest.json"
HUMAN_LABELS = ROOT / "human_full_semantic_boundary_rescue_v1" / "labels_resolved.json"
HARD = ROOT / "human_full_semantic_boundary_rescue_v1" / "hard_exclusion_intervals.json"
WORDS_DIR = ROOT / "full_source_local_word_scores_v1" / "source_reports"
OUT = ROOT / "full_semantic_boundary_rescue_round2_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RANDOM_SEED = 202608021802

# parent, variant, first word, last word, boundary override
TASKS = [
    ("R004", "restore_missing_mou", 1689, 1706, "word_aware"),
    ("R001", "duration_safe_under_6s", 2514, 2536, "trim_leading_silence"),
    ("R015", "long_sentence_part_1", 1521, 1543, "tight_40ms_25ms"),
    ("R015", "long_sentence_part_2", 1544, 1560, "word_aware"),
    ("R008", "environment_check_part_1", 213, 222, "word_aware"),
    ("R008", "environment_check_part_2", 223, 241, "word_aware"),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def custom_bounds(
    source: Path, words: list[dict], left: int, right: int, mode: str
) -> tuple[float, float, dict]:
    start, end, details = base.word_aware_bounds(source, words, left, right)
    if mode == "trim_leading_silence":
        start = max(0.0, float(words[left]["start"]) - 0.035)
        end = float(words[right]["end"]) + 0.025
        details = {**details, "override": mode}
    elif mode == "tight_40ms_25ms":
        start = max(0.0, float(words[left]["start"]) - 0.040)
        end = float(words[right]["end"]) + 0.025
        details = {**details, "override": mode}
    if end - start > 6.0005:
        raise RuntimeError(f"Duration still exceeds 6s: {left}-{right} {end-start:.4f}")
    return start, end, details


def main() -> None:
    parents = {
        str(row["blind_id"]): row
        for row in json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))["samples"]
    }
    labels = {
        str(row["blind_id"]): row
        for row in json.loads(HUMAN_LABELS.read_text(encoding="utf-8"))["samples"]
    }
    hard = json.loads(HARD.read_text(encoding="utf-8"))["intervals"]
    candidates = []
    for parent_id, variant, left, right, mode in TASKS:
        parent = parents[parent_id]
        human = labels[parent_id]
        source_id = str(parent["source_id"])
        source = Path(parent["source"]).resolve()
        words = json.loads(
            (WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8")
        )["word_rows"]
        start, end, boundary = custom_bounds(source, words, left, right, mode)
        hard_hits = [
            row["reason"]
            for row in hard
            if str(row["source_id"]) == source_id
            and overlaps(start, end, float(row["start"]), float(row["end"]))
        ]
        if hard_hits:
            raise RuntimeError(f"{parent_id}/{variant} overlaps hard mask: {hard_hits}")
        raw, rate = base.read_mono(source, start, end)
        clip = base.micro_fade(audio_base.resample_1d(raw, rate, OUTPUT_SR))
        selected_words = words[left : right + 1]
        text = "".join(str(row.get("text") or "") for row in selected_words)
        candidates.append(
            {
                "kind": "full_semantic_boundary_rescue_round2",
                "parent_blind_id": parent_id,
                "root_origin_blind_id": str(parent["origin_blind_id"]),
                "source_id": source_id,
                "source": str(source),
                "parent_human_class": str(human["final_class"]),
                "parent_human_note": str(human["human_note"]),
                "variant": variant,
                "start": start,
                "end": end,
                "duration": end - start,
                "word_start_index": left,
                "word_end_index": right,
                "text": text,
                "boundary": boundary,
                "hard_mask_hits": hard_hits,
                "_clip": clip,
            }
        )

    random.Random(RANDOM_SEED).shuffle(candidates)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(candidates, 1):
        blind_id = f"T{index:03d}"
        clip = row.pop("_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if not (1.5 <= info.duration <= 6.0005):
            raise RuntimeError(f"Invalid review duration: {output} {info.duration}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(clip)

    source_stats: dict[str, dict] = defaultdict(lambda: {"samples": 0, "duration": 0.0})
    for row in candidates:
        source_stats[row["source_id"]]["samples"] += 1
        source_stats[row["source_id"]]["duration"] += row["duration"]
    audit = {
        "version": "full_semantic_boundary_rescue_round2_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "restore_missing_head": "include the adjacent soft-fail word omitted by the first rescue",
            "long_passes": "re-cut into <=6s word-complete pieces; each new boundary must be heard",
            "quality_failure": "split only to test whether either subsection is free of the reported environment sound",
            "source_audio": "original BS-Roformer vocals; no reconstruction/denoise",
            "status": "human review only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(candidates),
        "source_stats": dict(sorted(source_stats.items())),
        "samples": candidates,
    }
    public = {
        "version": "full_semantic_boundary_rescue_round2_v1_blind",
        "sample_count": len(candidates),
        "samples": [
            {
                "blind_id": row["blind_id"],
                "audio": row["audio_rel"],
                "duration_sec": round(row["output_duration"], 3),
            }
            for row in candidates
        ],
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    write_json(
        OUT / "report.json",
        {
            "version": "full_semantic_boundary_rescue_round2_v1_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "samples": len(candidates),
            "origins": len({row["root_origin_blind_id"] for row in candidates}),
            "total_duration_sec": sum(row["duration"] for row in candidates),
            "hard_mask_overlap_count": 0,
            "training_started": False,
            "existing_training_lists_modified": False,
            "score_90_claimed": False,
        },
    )
    (OUT / "README_试听说明.txt").write_text(
        "只剩 6 条最终复核：1 条补回缺失开头，2 条是已通过长句的训练长度拆分，2 条检查环境音是否只污染局部，另 1 条是 6.075 秒通过句的安全缩短。/n"
        "仍按最严格标准：整条伊蕾娜、首尾完整、无环境音/杂音才不勾错误。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text(
        '@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "samples": len(candidates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
