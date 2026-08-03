# -*- coding: utf-8 -*-
"""Build precise rescues for six salvageable v2 acceptance failures."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
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
HARD_MASKS = ROOT / "full_semantic_mining_v2" / "hard_exclusion_masks.json"
OUT = ROOT / "full_semantic_acceptance_v2_rescue_v1"
AUDIO_DIR = OUT / "audio"

OUTPUT_SR = 32000
RANDOM_SEED = 202608021846

# Complete-word rescues. B005 also includes the complete >6s sentence so the
# listener can judge identity/continuity before the two train-length clauses.
WORD_TASKS = [
    ("B005", "complete_sentence_reference", 615, 648),
    ("B005", "complete_clause_1", 615, 625),
    ("B005", "complete_clause_2", 626, 648),
    ("B007", "restore_missing_toku", 770, 779),
    ("B022", "restore_question_ka", 1232, 1240),
]

# Millisecond transition rescues. The source end stays at the already-heard
# safe end; only the contaminated seamless start changes.
MANUAL_TASKS = [
    ("B013", "start_plus_20ms", 413.790, 419.160, 1479, 1497),
    ("B013", "start_plus_40ms", 413.810, 419.160, 1479, 1497),
    ("B013", "start_plus_60ms", 413.830, 419.160, 1479, 1497),
    ("B024", "target_reply_exact", 956.610, 959.330, 3292, 3297),
    ("B024", "target_reply_plus_20ms", 956.630, 959.330, 3292, 3297),
    ("B024", "target_reply_plus_40ms", 956.650, 959.330, 3292, 3297),
    ("B009", "second_reply_exact", 210.380, 212.050, 644, 647),
    ("B009", "second_reply_plus_20ms", 210.400, 212.050, 644, 647),
    ("B009", "second_reply_plus_40ms", 210.420, 212.050, 644, 647),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def overlaps(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def main() -> None:
    labels = {
        str(row["blind_id"]): row
        for row in json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    }
    hard_masks = json.loads(HARD_MASKS.read_text(encoding="utf-8"))
    expected = {"B005", "B007", "B009", "B013", "B022", "B024"}
    actual = {
        blind_id
        for blind_id, row in labels.items()
        if row["role"] == "machine_candidate_v2"
        and row["final_class"] in {"boundary_cut", "mixed"}
    }
    if actual != expected:
        raise RuntimeError(f"Rescue origin mismatch: expected={sorted(expected)} actual={sorted(actual)}")

    word_cache: dict[str, list[dict]] = {}

    def words_for(source_id: str) -> list[dict]:
        if source_id not in word_cache:
            word_cache[source_id] = json.loads(
                (WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8")
            )["word_rows"]
        return word_cache[source_id]

    candidates = []

    def append_candidate(
        origin_id: str,
        variant: str,
        start: float,
        end: float,
        left: int,
        right: int,
        boundary: dict,
    ) -> None:
        human = labels[origin_id]
        source_id = str(human["source_id"])
        source = Path(human["source"]).resolve()
        words = words_for(source_id)
        if not (0 <= left <= right < len(words)):
            raise IndexError(f"Invalid words for {origin_id}/{variant}: {left}-{right}")
        duration = end - start
        if duration < 1.5 or duration > 7.0:
            raise RuntimeError(f"Invalid rescue duration {origin_id}/{variant}: {duration:.3f}")
        hits = [
            mask["reason"]
            for mask in hard_masks.get(source_id, [])
            if overlaps(start, end, float(mask["start"]), float(mask["end"]))
        ]
        if hits:
            raise RuntimeError(f"{origin_id}/{variant} overlaps existing hard masks: {hits}")
        raw, rate = base.read_mono(source, start, end)
        clip = base.micro_fade(audio_base.resample_1d(raw, rate, OUTPUT_SR))
        selected_words = words[left : right + 1]
        text = "".join(str(word.get("text") or "") for word in selected_words)
        if not text.strip() or "/ufffd" in text or "|" in text:
            raise RuntimeError(f"Invalid text for {origin_id}/{variant}: {text!r}")
        candidates.append(
            {
                "kind": "full_semantic_acceptance_v2_rescue",
                "origin_blind_id": origin_id,
                "origin_class": str(human["final_class"]),
                "origin_note": str(human["human_note"]),
                "origin_start": float(human["start"]),
                "origin_end": float(human["end"]),
                "origin_text": str(human["text"]),
                "source_id": source_id,
                "source": str(source),
                "variant": variant,
                "start": float(start),
                "end": float(end),
                "duration": float(duration),
                "word_start_index": left,
                "word_end_index": right,
                "text": text,
                "boundary": boundary,
                "hard_mask_hits": hits,
                "_clip": clip,
            }
        )

    for origin_id, variant, left, right in WORD_TASKS:
        human = labels[origin_id]
        source_id = str(human["source_id"])
        source = Path(human["source"]).resolve()
        words = words_for(source_id)
        start, end, details = base.word_aware_bounds(source, words, left, right)
        append_candidate(origin_id, variant, start, end, left, right, details)

    for origin_id, variant, start, end, left, right in MANUAL_TASKS:
        words = words_for(str(labels[origin_id]["source_id"]))
        append_candidate(
            origin_id,
            variant,
            start,
            end,
            left,
            right,
            {
                "mode": "human_transition_millisecond_sweep",
                "first_target_word_start": float(words[left]["start"]),
                "last_target_word_end": float(words[right]["end"]),
                "start_inside_first_word_sec": max(0.0, start - float(words[left]["start"])),
            },
        )

    random.Random(RANDOM_SEED).shuffle(candidates)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(candidates, 1):
        blind_id = f"C{index:03d}"
        clip = row.pop("_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        info = sf.info(str(output))
        if info.samplerate != OUTPUT_SR or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(f"Invalid output format: {output}")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = float(info.duration)
        row["wave"] = page_base.waveform(np.asarray(clip, dtype=np.float32))

    origin_counts = Counter(row["origin_blind_id"] for row in candidates)
    source_stats: dict[str, dict] = defaultdict(lambda: {"samples": 0, "duration": 0.0})
    for row in candidates:
        source_stats[row["source_id"]]["samples"] += 1
        source_stats[row["source_id"]]["duration"] += float(row["duration"])
    audit = {
        "version": "full_semantic_acceptance_v2_rescue_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "original BS-Roformer vocals only",
            "boundary": "restore missing full words or sweep the exact seamless transition by 20ms",
            "mixed": "remove only the contaminated prefix; never reconstruct or denoise the target voice",
            "quality_B018": "not included; footstep-contaminated 1.635s clip remains excluded",
            "status": "blind human review only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "origin_count": len(origin_counts),
        "sample_count": len(candidates),
        "origin_variant_counts": dict(sorted(origin_counts.items())),
        "samples": candidates,
    }
    report = {
        "version": "full_semantic_acceptance_v2_rescue_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "origins": len(origin_counts),
        "samples": len(candidates),
        "duration_sec": sum(float(row["duration"]) for row in candidates),
        "longer_than_6s": sum(float(row["duration"]) > 6.0 for row in candidates),
        "origin_variant_counts": dict(sorted(origin_counts.items())),
        "source_stats": dict(sorted(source_stats.items())),
        "hard_mask_overlap_count": 0,
        "quality_clip_B018_excluded": True,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "private_manifest.json", audit)
    write_json(OUT / "report.json", report)
    (OUT / "README_试听说明.txt").write_text(
        "这 14 条来自 6 个可救问题；脚步声 B018 已排除，不在本页。/n"
        "同一问题会有多个盲版本。只要某版整条仅伊蕾娜、开头结尾完整且无杂音，就让它通过。/n"
        "仍有别人声音请选“混入/无缝接话”；只切字请选具体的开头/结尾类型。/n"
        "B005 有一个超过 6 秒的完整句对照和两个训练长度分句，新切出的每条都必须独立判断。/n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
