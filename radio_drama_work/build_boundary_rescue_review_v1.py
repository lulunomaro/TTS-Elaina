# -*- coding: utf-8 -*-
"""Build the small human review pack containing the best rescue per origin."""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

import build_boundary_rescue_all_v1 as rescue
import build_human_calibration_v1 as page_base


POOL = ROOT / "boundary_rescue_all_v5" / "audit_manifest.json"
LABELS = ROOT / "human_calibration_combined_v3" / "labels_resolved.json"
MANIFEST = ROOT / "full_track_manifest_v1.json"
WORDS_DIR = ROOT / "word_target_all_v1" / "source_reports"
OUT = ROOT / "boundary_rescue_review_v1"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RANDOM_SEED = 20260803

# A single best version per source interval.  The three explicit word ranges
# replace incomplete machine extensions with a semantically complete sentence.
SELECTIONS = [
    {"origin": "round2:R028", "variant": "phrase_context"},
    {"origin": "round2:R030", "variant": "phrase_context"},
    {"origin": "round2:R034", "variant": "tight_word_complete"},
    {"origin": "round2:R038", "variant": "tight_word_complete"},
    {"origin": "round2:R039", "word_range": (527, 542), "reason": "complete ご心配なく…読んでますよ"},
    {"origin": "round2:R042", "variant": "phrase_context"},
    {"origin": "round2:R045", "word_range": (288, 322), "reason": "complete 再び味見…なっていました"},
    {"origin": "round2:R046", "word_range": (281, 310), "reason": "stop before the next S-speaker utterance"},
    {"origin": "round3:A001", "variant": "tight_word_complete"},
    {"origin": "round3:A012", "variant": "tight_word_complete"},
    {"origin": "round3:A013", "variant": "phrase_context"},
    {"origin": "round3:A019", "variant": "tight_word_complete"},
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def resample_if_needed(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == OUTPUT_SR:
        return audio
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(audio), sr, OUTPUT_SR).numpy()


def main() -> None:
    pool_rows = json.loads(POOL.read_text(encoding="utf-8"))["samples"]
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    labels_by_key = {str(row["human_key"]): row for row in labels}
    tracks = {str(row["id"]): row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["tracks"]}
    selected = []
    audio_cache: dict[str, tuple[np.ndarray, int, np.ndarray, np.ndarray]] = {}

    for choice in SELECTIONS:
        origin = str(choice["origin"])
        human = labels_by_key[origin]
        if "variant" in choice:
            matches = [
                row
                for row in pool_rows
                if str(row["origin_human_key"]) == origin and str(row["variant"]) == str(choice["variant"])
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one pool match for {choice}, got {len(matches)}")
            row = dict(matches[0])
            clip, sr = sf.read(str(row["audio"]), dtype="float32", always_2d=True)
            row["_audio_clip"] = resample_if_needed(clip.mean(axis=1).astype(np.float32), sr)
            row["selection_reason"] = "best machine rescue variant"
            selected.append(row)
            continue

        source_id = str(human["source_id"])
        source_path = str(Path(tracks[source_id]["vocals"]).resolve())
        if source_id not in audio_cache:
            source_audio, sr = sf.read(source_path, dtype="float32", always_2d=True)
            mono = source_audio.mean(axis=1).astype(np.float32, copy=False)
            times, rms = rescue.frame_rms(mono, sr)
            audio_cache[source_id] = (mono, sr, times, rms)
        mono, sr, times, rms = audio_cache[source_id]
        words = json.loads((WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8"))["word_rows"]
        left, right = (int(value) for value in choice["word_range"])
        context = np.flatnonzero(
            (times >= float(words[left]["start"]) - 1.0) & (times <= float(words[right]["end"]) + 1.0)
        )
        speech_level = float(np.quantile(rms[context], 0.80)) if len(context) else float(np.quantile(rms, 0.80))
        left_boundary = rescue.start_boundary(words, left, times, rms, speech_level)
        right_boundary = rescue.end_boundary(words, right, times, rms, speech_level)
        # Keep only a short natural pre/post roll even when the neighbouring
        # ASR gap is very long; long silence is not useful training material.
        start = max(float(left_boundary["time"]), float(words[left]["start"]) - 0.22)
        end = min(float(right_boundary["time"]), float(words[right]["end"]) + 0.25)
        source_rows = [row for row in labels if str(row.get("source_id")) == source_id]
        hard_hits = rescue.overlaps_unsafe(start, end, source_rows, origin)
        if hard_hits:
            raise RuntimeError(f"Manual rescue {origin} overlaps hard intervals: {hard_hits}")
        model = rescue.load_model(source_id)
        fraction = rescue.target_fraction(model, start, end)
        if fraction is not None and fraction < 0.50:
            raise RuntimeError(f"Manual rescue {origin} has unsafe target fraction {fraction}")
        start_sample = max(0, round(start * sr))
        end_sample = min(len(mono), round(end * sr))
        clip = resample_if_needed(mono[start_sample:end_sample].copy(), sr)
        selected.append(
            {
                "kind": "boundary_rescue_candidate",
                "origin_human_key": origin,
                "origin_blind_id": human["blind_id"],
                "origin_round": human["human_round"],
                "source_id": source_id,
                "source": source_path,
                "origin_start": float(human["start"]),
                "origin_end": float(human["end"]),
                "variant": "manual_semantic_complete",
                "start": start,
                "end": end,
                "duration": end - start,
                "word_start_index": left,
                "word_end_index": right,
                "text": "".join(str(words[idx].get("text", "")) for idx in range(left, right + 1)),
                "target_cluster_fraction": fraction,
                "left_boundary": left_boundary,
                "right_boundary": right_boundary,
                "hard_overlap": hard_hits,
                "selection_reason": str(choice["reason"]),
                "_audio_clip": clip,
            }
        )

    if len(selected) != len(SELECTIONS) or len({row["origin_human_key"] for row in selected}) != len(selected):
        raise RuntimeError("Final rescue selection is missing or duplicates an origin")
    random.Random(RANDOM_SEED).shuffle(selected)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, 1):
        blind_id = f"C{idx:03d}"
        clip = row.pop("_audio_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = len(clip) / OUTPUT_SR
        row["wave"] = page_base.waveform(clip)

    audit = {
        "version": "boundary_rescue_review_v1_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source": "original full BS-Roformer vocals",
            "scope": "one best rescue for each of 12 speaker-plausible boundary origins",
            "excluded": "six historical boundary origins rejected by corrected source-local speaker clusters",
            "manual": "three incomplete machine extensions replaced with complete ASR semantic spans",
            "hard_overlap": "zero overlap with known non-target/mixed/quality intervals other than the rescued origin",
            "status": "human review only; not training data",
        },
        "sample_count": len(selected),
        "samples": selected,
    }
    public = {
        "version": "boundary_rescue_review_v1_blind",
        "sample_count": len(selected),
        "samples": [
            {"blind_id": row["blind_id"], "audio": row["audio_rel"], "duration_sec": round(row["output_duration"], 3)}
            for row in selected
        ],
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    (OUT / "README_试听说明.txt").write_text(
        "这是切头切尾素材从整轨重新裁切后的救援验收，共 12 条。/n"
        "完整、纯伊蕾娜且自然才不勾错误；仍有半字、半句话、其他角色或杂音都要勾错误。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text('@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8")
    print(json.dumps({"output": str(OUT), "samples": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
