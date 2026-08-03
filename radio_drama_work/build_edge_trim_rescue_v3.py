# -*- coding: utf-8 -*-
"""Second inward trim: start at the next semantically complete target phrase."""
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
AUDIT = ROOT / "edge_trim_rescue_v2" / "audit_manifest.json"
LABELS = ROOT / "human_edge_trim_rescue_v2" / "labels_resolved.json"
WORDS_DIR = ROOT / "word_target_all_v1" / "source_reports"
OUT = ROOT / "edge_trim_rescue_v3"
AUDIO_DIR = OUT / "audio"
OUTPUT_SR = 32000
FADE_SEC = 0.005
RANDOM_SEED = 20260805

# Parent blind ID -> first complete target word after the residual edge voice.
START_WORD = {
    "D001": 345,  # この後の展開と言われても… (drop the non-target はい/いや edge)
    "D002": 106,  # 今日は魔女らしく… (drop the seamless preceding です/というわけで edge)
    "D005": 531,  # 私これでも… (drop the preceding けど/ご心配なく edge)
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8")


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == OUTPUT_SR:
        return audio
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(audio), sr, OUTPUT_SR).numpy()


def micro_fade(audio: np.ndarray) -> np.ndarray:
    count = min(len(audio) // 2, max(1, round(OUTPUT_SR * FADE_SEC)))
    ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
    audio[:count] *= ramp
    audio[-count:] *= ramp[::-1]
    return audio


def main() -> None:
    audit_rows = {row["blind_id"]: row for row in json.loads(AUDIT.read_text(encoding="utf-8"))["samples"]}
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    failures = [row for row in labels if row["final_class"] == "mixed"]
    if {str(row["blind_id"]) for row in failures} != set(START_WORD):
        raise RuntimeError("Unexpected edge-trim failures")

    cache: dict[str, tuple[np.ndarray, int]] = {}
    selected = []
    for human in failures:
        parent_id = str(human["blind_id"])
        parent = audit_rows[parent_id]
        source_id = str(parent["source_id"])
        words = json.loads((WORDS_DIR / f"{source_id}.json").read_text(encoding="utf-8"))["word_rows"]
        first = int(START_WORD[parent_id])
        last = int(parent["word_end_index"])
        start = float(words[first]["start"])
        end = min(float(parent["end"]), float(words[last]["end"]) + 0.18)
        if end - start < 1.50:
            raise RuntimeError(f"Trimmed clip too short: {parent_id}")
        source_path = str(Path(parent["source"]).resolve())
        if source_path not in cache:
            audio, sr = sf.read(source_path, dtype="float32", always_2d=True)
            cache[source_path] = (audio.mean(axis=1).astype(np.float32, copy=False), sr)
        mono, sr = cache[source_path]
        clip = mono[max(0, round(start * sr)) : min(len(mono), round(end * sr))].copy()
        clip = micro_fade(resample(clip, sr))
        selected.append(
            {
                "kind": "semantic_inward_trim_candidate",
                "parent_blind_id": parent_id,
                "root_origin_human_key": parent["origin_human_key"],
                "source_id": source_id,
                "source": source_path,
                "parent_start": float(parent["start"]),
                "parent_end": float(parent["end"]),
                "start": start,
                "end": end,
                "duration": end - start,
                "word_start_index": first,
                "word_end_index": last,
                "text": "".join(str(words[idx].get("text", "")) for idx in range(first, last + 1)),
                "trimmed_start_from_parent_sec": start - float(parent["start"]),
                "fade_sec": FADE_SEC,
                "strategy": "drop the contaminated transition phrase and begin at the next complete target phrase",
                "_audio_clip": clip,
            }
        )

    random.Random(RANDOM_SEED).shuffle(selected)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, 1):
        blind_id = f"E{idx:03d}"
        clip = row.pop("_audio_clip")
        output = AUDIO_DIR / f"{blind_id}.wav"
        sf.write(str(output), clip, OUTPUT_SR, subtype="PCM_16")
        row["blind_id"] = blind_id
        row["audio"] = str(output.resolve())
        row["audio_rel"] = f"audio/{output.name}"
        row["output_duration"] = len(clip) / OUTPUT_SR
        row["wave"] = page_base.waveform(clip)

    audit = {
        "version": "edge_trim_rescue_v3_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "input": "three clips still containing a tiny non-target start after exact edge trimming",
            "strategy": "discard the contaminated transition phrase and start from the next complete target phrase",
            "audio": "original full vocals plus a 5 ms click-prevention fade only",
            "status": "human review only; not training data",
            "local_resources": RESOURCE_POLICY,
        },
        "sample_count": len(selected),
        "samples": selected,
    }
    public = {
        "version": "edge_trim_rescue_v3_blind",
        "sample_count": len(selected),
        "samples": [
            {"blind_id": row["blind_id"], "audio": row["audio_rel"], "duration_sec": round(row["output_duration"], 3)}
            for row in selected
        ],
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "blind_manifest.json", public)
    (OUT / "README_试听说明.txt").write_text(
        "这 3 条已舍弃残留非目标声所在的过渡短语，改从下一处完整目标短句开始。/n"
        "重点检查第一个字是否完整、开头是否还残留别人；有问题继续勾错误。/n",
        encoding="utf-8",
    )
    (OUT / "开始试听.cmd").write_text('@echo off/r/nstart "" "%~dp0index.html"/r/n', encoding="utf-8")
    print(json.dumps({"output": str(OUT), "samples": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
