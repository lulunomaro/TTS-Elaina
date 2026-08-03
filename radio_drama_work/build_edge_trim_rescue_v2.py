# -*- coding: utf-8 -*-
"""Trim the five user-localized edge contaminations at exact ASR word boundaries."""
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

import build_human_calibration_v1 as page_base
from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)
AUDIT = ROOT / "boundary_rescue_review_v1" / "audit_manifest.json"
LABELS = ROOT / "human_boundary_rescue_v1" / "labels_resolved.json"
WORDS_DIR = ROOT / "word_target_all_v1" / "source_reports"
OUT = ROOT / "edge_trim_rescue_v2"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
RANDOM_SEED = 20260804
FADE_SEC = 0.005


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == OUTPUT_SR:
        return audio
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(audio), sr, OUTPUT_SR).numpy()


def apply_micro_fade(audio: np.ndarray, sr: int) -> np.ndarray:
    count = min(len(audio) // 2, max(1, round(sr * FADE_SEC)))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        audio[:count] *= ramp
        audio[-count:] *= ramp[::-1]
    return audio


def main() -> None:
    audit_rows = {row["blind_id"]: row for row in json.loads(AUDIT.read_text(encoding="utf-8"))["samples"]}
    label_rows = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    failures = [row for row in label_rows if row["final_class"] == "mixed"]
    if len(failures) != 5:
        raise RuntimeError(f"Expected five edge-contaminated rescues, got {len(failures)}")

    source_cache: dict[str, tuple[np.ndarray, int]] = {}
    selected = []
    for human in failures:
        source = audit_rows[str(human["blind_id"])]
        source_id = str(source["source_id"])
        words = json.loads((WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8"))["word_rows"]
        first = int(source["word_start_index"])
        last = int(source["word_end_index"])
        note = str(human.get("human_note") or "")
        trim_start = any(token in note for token in ("开头", "開頭", "起始", "前面"))
        trim_end = any(token in note for token in ("结尾", "結尾", "结束", "末尾", "后面"))
        if not trim_start and not trim_end:
            raise RuntimeError(f"No localized edge in note for {human['blind_id']}: {note}")

        start = float(words[first]["start"]) if trim_start else float(source["start"])
        end = float(words[last]["end"]) + 0.045 if trim_end else min(
            float(source["end"]), float(words[last]["end"]) + 0.22
        )
        if end - start < 1.50:
            raise RuntimeError(f"Trimmed sample is too short: {human['blind_id']} {start}-{end}")

        source_path = str(Path(source["source"]).resolve())
        if source_path not in source_cache:
            audio, sr = sf.read(source_path, dtype="float32", always_2d=True)
            source_cache[source_path] = (audio.mean(axis=1).astype(np.float32, copy=False), sr)
        mono, sr = source_cache[source_path]
        start_sample = max(0, round(start * sr))
        end_sample = min(len(mono), round(end * sr))
        clip = resample(mono[start_sample:end_sample].copy(), sr)
        clip = apply_micro_fade(clip, OUTPUT_SR)
        selected.append(
            {
                "kind": "edge_trim_rescue_candidate",
                "parent_blind_id": source["blind_id"],
                "origin_human_key": source["origin_human_key"],
                "source_id": source_id,
                "source": source_path,
                "parent_start": float(source["start"]),
                "parent_end": float(source["end"]),
                "start": start,
                "end": end,
                "duration": end - start,
                "word_start_index": first,
                "word_end_index": last,
                "text": "".join(str(words[idx].get("text", "")) for idx in range(first, last + 1)),
                "trim_start": trim_start,
                "trim_end": trim_end,
                "trimmed_start_sec": start - float(source["start"]),
                "trimmed_end_sec": float(source["end"]) - end,
                "fade_sec": FADE_SEC,
                "parent_feedback_note": note,
                "_audio_clip": clip,
            }
        )

    random.Random(RANDOM_SEED).shuffle(selected)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, 1):
        blind_id = f"D{idx:03d}"
        clip = row.pop("_audio_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = len(clip) / OUTPUT_SR
        row["wave"] = page_base.waveform(clip)

    audit = {
        "version": "edge_trim_rescue_v2_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "input": "the five rescue clips with user-localized edge-only non-target contamination",
            "start": "crop at the first intended ASR word onset when the user marked start contamination",
            "end": "crop 45 ms after the last intended ASR word when the user marked end contamination",
            "fade": "5 ms micro-fade to prevent a click; no denoise, separation, or voice reconstruction",
            "status": "human review only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(selected),
        "samples": selected,
    }
    public = {
        "version": "edge_trim_rescue_v2_blind",
        "sample_count": len(selected),
        "samples": [
            {"blind_id": row["blind_id"], "audio": row["audio_rel"], "duration_sec": round(row["output_duration"], 3)}
            for row in selected
        ],
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    (OUT / "README_试听说明.txt").write_text(
        "这 5 条只向内裁掉了你指出的边缘非目标声，并加 5ms 防点击淡入淡出。/n"
        "若仍听见哪怕一点其他角色，或新边界切掉了目标字，继续勾错误并写明开头/结尾。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text('@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8")
    print(json.dumps({"output": str(OUT), "samples": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
