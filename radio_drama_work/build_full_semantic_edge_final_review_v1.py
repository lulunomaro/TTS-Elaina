# -*- coding: utf-8 -*-
"""Create three millisecond-level start trims for the final A021 edge."""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import soundfile as sf

import build_full_semantic_acceptance_boundary_rescue_v1 as base
import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


PARENT_AUDIT = ROOT / "full_semantic_boundary_rescue_round2_v1" / "audit_manifest.json"
HARD = ROOT / "human_verified_semantic_rescues_v2" / "hard_exclusion_intervals.json"
OUT = ROOT / "full_semantic_edge_final_review_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RANDOM_SEED = 202608021815

# The failed clip started at 475.565.  The preceding ASR word ends and the
# intended 「もう」 begins at 475.590.  Test the exact boundary and two tiny
# inward trims so the listener can choose between residual vowel and cut /m/.
START_VARIANTS = [
    ("at_asr_target_onset", 475.590),
    ("target_onset_plus_20ms", 475.610),
    ("target_onset_plus_40ms", 475.630),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def main() -> None:
    parent = next(
        row
        for row in json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))["samples"]
        if row["blind_id"] == "T002"
    )
    hard = json.loads(HARD.read_text(encoding="utf-8"))["intervals"]
    source = Path(parent["source"]).resolve()
    end = float(parent["end"])
    candidates = []
    for variant, start in START_VARIANTS:
        hits = [
            row["reason"]
            for row in hard
            if str(row["source_id"]) == str(parent["source_id"])
            and overlaps(start, end, float(row["start"]), float(row["end"]))
        ]
        if hits:
            raise RuntimeError(f"Final edge variant overlaps hard mask: {hits}")
        raw, rate = base.read_mono(source, start, end)
        clip = base.micro_fade(audio_base.resample_1d(raw, rate, OUTPUT_SR))
        candidates.append(
            {
                "kind": "full_semantic_edge_final_review",
                "parent_blind_id": "T002",
                "root_origin_blind_id": "A021",
                "source_id": str(parent["source_id"]),
                "source": str(source),
                "variant": variant,
                "start": start,
                "end": end,
                "duration": end - start,
                "removed_from_failed_parent_sec": start - float(parent["start"]),
                "word_start_index": int(parent["word_start_index"]),
                "word_end_index": int(parent["word_end_index"]),
                "text": str(parent["text"]),
                "hard_mask_hits": hits,
                "_clip": clip,
            }
        )

    random.Random(RANDOM_SEED).shuffle(candidates)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(candidates, 1):
        blind_id = f"U{index:03d}"
        clip = row.pop("_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(clip)

    audit = {
        "version": "full_semantic_edge_final_review_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "problem": "A021 content is clean after restoring もう, but a tiny preceding vowel remains",
            "variants": "target ASR onset, +20ms, +40ms",
            "source_audio": "original BS-Roformer vocals; 2ms micro-fade only",
            "status": "human review only; choose any no-error version, prefer the earliest safe start",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(candidates),
        "samples": candidates,
    }
    public = {
        "version": "full_semantic_edge_final_review_v1_blind",
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
            "version": "full_semantic_edge_final_review_v1_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "samples": len(candidates),
            "origin": "A021",
            "training_started": False,
            "existing_training_lists_modified": False,
            "score_90_claimed": False,
        },
    )
    (OUT / "README_试听说明.txt").write_text(
        "最后只听 3 条：内容相同，仅开头分别多裁 25ms、45ms、65ms。/n"
        "选择没有残留‘诶’且‘もう’没有被切头的版本；满足时不勾错误。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text(
        '@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "samples": len(candidates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
