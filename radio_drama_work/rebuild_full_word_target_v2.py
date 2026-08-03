# -*- coding: utf-8 -*-
"""Rebuild the final candidate set from a completed full-pass report.

This intentionally reuses the expensive ASR/embedding results from v1 and
only recalibrates gates that were checked against the four user-confirmed
positive clips.  It writes a new output version and leaves v1 untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_word_target_extract as x  # noqa: E402


PREVIOUS_REPORT = x.ROOT / "full_word_target_v1" / "report.json"
OUT = x.ROOT / "full_word_target_v2"
x.OUT = OUT
x.ACCEPTED_DIR = OUT / "accepted_segments"
x.REVIEW_DIR = OUT / "review"
x.REJECTED_DIR = OUT / "rejected"
x.REPORT_PATH = OUT / "report.json"
x.VALIDATION_PATH = OUT / "validation.json"
x.CANDIDATE_LIST = x.ROOT / "full_word_target_train_candidate.list"


def rebuild() -> None:
    previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    forbidden_by_source = x.parse_forbidden_intervals()
    OUT.mkdir(parents=True, exist_ok=True)
    x.ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    x.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    x.REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "full_word_target_v2",
        "note": "Recalibrated high-precision word-level target gate; reuses v1 ASR and embeddings, outputs original separated vocal waveform.",
        "inputs": previous.get("inputs", []),
        "excluded_basenames": sorted(x.EXCLUDED_BASENAMES),
        "calibration": {
            "based_on": str(PREVIOUS_REPORT),
            "reason": "v1 direct/frame gates rejected all words; four user-confirmed pure positives were used to verify the lower direct gate and cluster-veto rule.",
        },
        "parameters": {
            "word_center_min": x.WORD_CENTER_MIN,
            "word_p25_min": x.WORD_P25_MIN,
            "word_margin_min": x.WORD_MARGIN_MIN,
            "cluster_vote_min": x.CLUSTER_VOTE_MIN,
            "cluster_margin_min": x.CLUSTER_MARGIN_MIN,
            "min_final_sec": x.MIN_FINAL_SEC,
            "merge_gap_sec": x.MERGE_GAP_SEC,
            "other_ratio_max": x.OTHER_RATIO_MAX,
        },
        "forbidden_intervals": forbidden_by_source,
        "files": [],
    }
    list_lines: list[str] = []

    for old_file in previous.get("files", []):
        source = Path(old_file["source"])
        if not source.exists():
            raise FileNotFoundError(source)
        vocal = x.load_audio_32(source)
        source_forbidden = forbidden_by_source.get(source.name, [])
        rows = [dict(row) for row in old_file.get("scored_words", [])]
        cluster_margin = float(old_file.get("cluster_margin", 0.0))
        for row in rows:
            row["forbidden"] = x.overlaps(row["start"], row["end"], source_forbidden)
            row["direct_pass"] = bool(
                row["target_center"] >= x.WORD_CENTER_MIN
                and row["target_p25"] >= x.WORD_P25_MIN
                and row["margin"] >= x.WORD_MARGIN_MIN
            )
            row["frame_pass"] = bool(
                row.get("cluster_vote", 0.0) >= x.CLUSTER_VOTE_MIN
                and cluster_margin >= x.CLUSTER_MARGIN_MIN
            )
            row["other_pass"] = bool(row.get("other_ratio", 0.0) <= x.OTHER_RATIO_MAX)
            row["accepted"] = bool(row["direct_pass"] and row["frame_pass"] and not row["forbidden"] and row["other_pass"])

        accepted = [row for row in rows if row["accepted"]]
        merged = x.merge_accepted(accepted, rows, source_forbidden)
        final_segments = []
        short_count = 0
        for segment in merged:
            if segment["end"] - segment["start"] < x.MIN_FINAL_SEC:
                short_count += 1
                continue
            if x.overlaps(segment["start"], segment["end"], source_forbidden):
                continue
            clip, start, end = x.clip_from_source(vocal, segment["start"], segment["end"])
            text = "".join(word["text"] for word in segment["words"])
            output = x.ACCEPTED_DIR / f"{source.stem}_target_{len(final_segments):04d}.wav"
            if output.name in x.EXCLUDED_BASENAMES:
                raise RuntimeError(f"Forbidden output name generated: {output.name}")
            sf.write(str(output), clip.numpy().T, x.SAMPLE_RATE, subtype="PCM_16")
            final_segments.append(
                {
                    "file": str(output),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text": text,
                    "word_count": len(segment["words"]),
                    "words": segment["words"],
                }
            )
            list_lines.append(f"{output.resolve()}|full_word_target_v2|JA|{text}")

        x.write_review(vocal, rows, source.name)
        reject_counts = {
            "confirmed_forbidden_interval": sum(bool(row.get("forbidden")) for row in rows if not row["accepted"]),
            "target_similarity_gate": sum(not row["direct_pass"] and not row.get("forbidden") for row in rows),
            "local_cluster_gate": sum(row["direct_pass"] and not row["frame_pass"] and not row.get("forbidden") for row in rows),
            "other_stem_dominant": sum(row["direct_pass"] and row["frame_pass"] and not row["other_pass"] for row in rows),
            "merged_segment_too_short": short_count,
        }
        x.REJECTED_DIR.joinpath(f"{source.stem}_rejected.json").write_text(
            json.dumps([row for row in rows if not row["accepted"]], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["files"].append(
            {
                "source": str(source),
                "other_source": old_file.get("other_source"),
                "duration": vocal.shape[1] / x.SAMPLE_RATE,
                "asr_word_count": old_file.get("asr_word_count", 0),
                "scored_word_count": len(rows),
                "accepted_word_count": len(accepted),
                "final_segment_count": len(final_segments),
                "final_duration": sum(segment["duration"] for segment in final_segments),
                "reject_counts": reject_counts,
                "cluster_margin": cluster_margin,
                "cluster_scores": old_file.get("cluster_scores", []),
                "segments": final_segments,
                "scored_words": rows,
            }
        )
        print(
            f"{source.name}: accepted_words={len(accepted)} final_segments={len(final_segments)} final_sec={sum(s['duration'] for s in final_segments):.1f}",
            flush=True,
        )

    x.CANDIDATE_LIST.write_text("/n".join(list_lines) + ("/n" if list_lines else ""), encoding="utf-8")
    report["candidate_count"] = len(list_lines)
    report["candidate_duration"] = sum(item["final_duration"] for item in report["files"])
    x.REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = x.validate()
    print(f"saved_report={x.REPORT_PATH}", flush=True)
    print(f"saved_list={x.CANDIDATE_LIST} count={len(list_lines)}", flush=True)
    print(f"validation_passed={validation['passed']}", flush=True)
    if not validation["passed"]:
        raise RuntimeError("Validation failed: " + str(x.VALIDATION_PATH))


if __name__ == "__main__":
    rebuild()
