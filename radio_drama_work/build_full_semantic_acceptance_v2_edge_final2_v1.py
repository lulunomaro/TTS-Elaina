# -*- coding: utf-8 -*-
"""Build the final tiny B013/B024 edge-only rescue review set."""
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

import numpy as np
import soundfile as sf

import build_full_semantic_acceptance_boundary_rescue_v1 as base
import build_human_calibration_v1 as page_base
import full_speaker_diarize_new_v1 as audio_base


LABELS = ROOT / "human_full_semantic_acceptance_v2" / "labels_resolved.json"
WORDS_DIR = ROOT / "full_source_local_word_scores_v1" / "source_reports"
HARD_MASK_FILES = (
    ROOT / "full_semantic_mining_v2" / "hard_exclusion_masks.json",
    ROOT
    / "human_full_semantic_acceptance_v2_rescue_v1"
    / "hard_exclusion_intervals.json",
)
OUT = ROOT / "full_semantic_acceptance_v2_edge_final2_v1"
AUDIO_DIR = OUT / "audio"

OUTPUT_SR = 32000
RANDOM_SEED = 202608021901

# The previous 20/40/60 ms sweeps were still inside the preceding speaker's
# phrase. These variants discard that entire phrase and begin at the first
# confirmed target word instead.
TASKS = (
    ("B013", "restart_at_sensei_exact", 414.950, 418.990, 1483, 1497),
    ("B013", "restart_at_sensei_plus_20ms", 414.970, 418.990, 1483, 1497),
    ("B024", "restart_at_kurisotsu_exact", 957.830, 959.330, 3293, 3297),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def load_hard_masks() -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}
    for path in HARD_MASK_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "intervals" in payload:
            rows = payload["intervals"]
            for row in rows:
                merged.setdefault(str(row["source_id"]), []).append(row)
        else:
            for source_id, rows in payload.items():
                merged.setdefault(str(source_id), []).extend(rows)
    return merged


def main() -> None:
    labels = {
        str(row["blind_id"]): row
        for row in json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    }
    hard_masks = load_hard_masks()
    word_cache: dict[str, list[dict]] = {}
    candidates: list[dict] = []

    for origin_id, variant, start, end, left, right in TASKS:
        origin = labels[origin_id]
        source_id = str(origin["source_id"])
        source = Path(origin["source"]).resolve()
        if source_id not in word_cache:
            word_cache[source_id] = json.loads(
                (WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8")
            )["word_rows"]
        words = word_cache[source_id]
        if not (0 <= left <= right < len(words)):
            raise IndexError(f"Invalid words for {origin_id}/{variant}: {left}-{right}")
        duration = end - start
        if not 1.5 <= duration <= 6.0:
            raise RuntimeError(f"Invalid duration for {origin_id}/{variant}: {duration}")
        hits = [
            row
            for row in hard_masks.get(source_id, [])
            if overlaps(start, end, float(row["start"]), float(row["end"]))
        ]
        if hits:
            raise RuntimeError(f"{origin_id}/{variant} overlaps hard masks: {hits}")
        selected_words = words[left : right + 1]
        text = "".join(str(row.get("text") or "") for row in selected_words)
        if not text.strip() or "/ufffd" in text or "|" in text:
            raise RuntimeError(f"Invalid text for {origin_id}/{variant}: {text!r}")
        raw, rate = base.read_mono(source, start, end)
        clip = base.micro_fade(audio_base.resample_1d(raw, rate, OUTPUT_SR))
        candidates.append(
            {
                "kind": "full_semantic_acceptance_v2_edge_final2",
                "origin_blind_id": origin_id,
                "origin_note": str(origin["human_note"]),
                "source_id": source_id,
                "source": str(source),
                "variant": variant,
                "start": start,
                "end": end,
                "duration": duration,
                "word_start_index": left,
                "word_end_index": right,
                "text": text,
                "boundary": {
                    "mode": "drop_entire_non_target_lead_phrase",
                    "first_word": str(words[left]["text"]),
                    "first_word_start": float(words[left]["start"]),
                    "last_word": str(words[right]["text"]),
                    "last_word_end": float(words[right]["end"]),
                    "start_inside_first_word_sec": max(
                        0.0, start - float(words[left]["start"])
                    ),
                    "micro_fade_sec": 0.002,
                },
                "hard_mask_hits": [],
                "_clip": clip,
            }
        )

    random.Random(RANDOM_SEED).shuffle(candidates)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(candidates, 1):
        blind_id = f"E{index:03d}"
        clip = row.pop("_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if (
            info.samplerate != OUTPUT_SR
            or info.channels != 1
            or info.subtype != "PCM_16"
            or not 1.5 <= info.duration <= 6.0
        ):
            raise RuntimeError(f"Invalid review output: {output} {info}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(np.asarray(clip, dtype=np.float32))

    audit = {
        "version": "full_semantic_acceptance_v2_edge_final2_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "original BS-Roformer vocals only",
            "purpose": "final three-clip human boundary review for B013/B024",
            "speaker_rule": "preceding speaker phrase is discarded in full",
            "status": "review only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(candidates),
        "samples": candidates,
    }
    report = {
        "version": "full_semantic_acceptance_v2_edge_final2_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(candidates),
        "duration_sec": sum(float(row["output_duration"]) for row in candidates),
        "origins": sorted({row["origin_blind_id"] for row in candidates}),
        "hard_mask_overlap_count": 0,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
