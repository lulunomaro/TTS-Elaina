# -*- coding: utf-8 -*-
"""Resumable GPU Japanese word-timestamp ASR for the 12 anime episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import torch
from faster_whisper import WhisperModel


MANIFEST = ROOT / "anime_batch_v1" / "manifest.json"
OUT = ROOT / "anime_mining_v1" / "asr"
CACHE = OUT / "source_reports"
MODEL_DIR = (
    ROOT.parent
    / "GPT-SoVITS-v2pro-20250604-nvidia50"
    / "GPT-SoVITS-v2pro-20250604-nvidia50"
    / "tools"
    / "asr"
    / "models"
    / "faster-whisper-large-v3-turbo"
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    temporary.replace(path)


def fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "model": str(MODEL_DIR.resolve()),
        "language": "ja",
        "beam_size": 5,
        "word_timestamps": True,
        "vad_min_silence_ms": 180,
        "condition_on_previous_text": False,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def validate_words(source_id: str, words: list[dict]) -> None:
    if not words:
        raise RuntimeError(f"{source_id}: ASR returned no words")
    previous_start = -1.0
    for index, word in enumerate(words):
        text = str(word.get("text") or "")
        start = float(word["start"])
        end = float(word["end"])
        if not text.strip() or "/ufffd" in text or "|" in text:
            raise RuntimeError(f"{source_id}: invalid ASR text at word {index}: {text!r}")
        if start < previous_start - 0.02 or end <= start:
            raise RuntimeError(
                f"{source_id}: invalid ASR timestamps at word {index}: {start}-{end}"
            )
        previous_start = start


def transcribe(model: WhisperModel, source: Path) -> list[dict]:
    segments, _ = model.transcribe(
        str(source),
        language="ja",
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 180},
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    words: list[dict] = []
    for segment_index, segment in enumerate(segments):
        for word in segment.words or []:
            text = str(word.word or "").strip()
            start = float(word.start)
            end = float(word.end)
            if text and end > start:
                words.append(
                    {
                        "text": text,
                        "start": start,
                        "end": end,
                        "probability": float(getattr(word, "probability", 0.0)),
                        "segment_index": segment_index,
                    }
                )
    return words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks = manifest.get("tracks", [])
    if len(tracks) != 12:
        raise RuntimeError(f"Expected 12 anime episodes, found {len(tracks)}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)

    CACHE.mkdir(parents=True, exist_ok=True)
    pending = []
    source_rows: list[dict] = []
    for track in tracks:
        source = Path(track["vocals"])
        if not source.is_file():
            raise FileNotFoundError(source)
        fp = fingerprint(source)
        cache = CACHE / f"{track['id']}.json"
        if args.resume and cache.is_file():
            prior = json.loads(cache.read_text(encoding="utf-8"))
            if prior.get("fingerprint") == fp and prior.get("words"):
                validate_words(str(track["id"]), prior["words"])
                source_rows.append(prior)
                print(
                    f"[{track['id']}] ASR resume words={len(prior['words'])}",
                    flush=True,
                )
                continue
        pending.append((track, source, fp, cache))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = None
    if pending:
        print(
            f"[model] faster-whisper large-v3-turbo device={device} "
            f"compute={compute_type} pending={len(pending)}",
            flush=True,
        )
        model = WhisperModel(
            str(MODEL_DIR),
            device=device,
            compute_type=compute_type,
            cpu_threads=2,
        )

    started = time.time()
    for track, source, fp, cache in pending:
        source_started = time.time()
        words = transcribe(model, source)
        validate_words(str(track["id"]), words)
        item = {
            "version": "anime_asr_v1_source",
            "fingerprint": fp,
            "source_id": str(track["id"]),
            "source": str(source.resolve()),
            "word_count": len(words),
            "elapsed_sec": time.time() - source_started,
            "words": words,
        }
        write_json(cache, item)
        source_rows.append(item)
        print(
            f"[{track['id']}] ASR done words={len(words)} "
            f"elapsed={item['elapsed_sec']:.1f}s",
            flush=True,
        )
        write_json(
            OUT / "report.json",
            {
                "version": "anime_asr_v1_partial",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "sources": sorted(source_rows, key=lambda row: row["source_id"]),
            },
        )

    source_rows.sort(key=lambda row: row["source_id"])
    report = {
        "version": "anime_asr_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "manifest": str(manifest_path),
        "model": str(MODEL_DIR),
        "device": device,
        "compute_type": compute_type,
        "local_resources": RESOURCE_POLICY,
        "coverage": {"expected": 12, "completed": len(source_rows)},
        "total_word_count": sum(len(row["words"]) for row in source_rows),
        "sources": source_rows,
        "training_started": False,
        "existing_training_lists_modified": False,
    }
    if len(source_rows) != 12:
        raise RuntimeError(f"Incomplete ASR coverage: {len(source_rows)}/12")
    write_json(OUT / "report.json", report)
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "total_word_count": report["total_word_count"],
                "report": str(OUT / "report.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
