# -*- coding: utf-8 -*-
"""Inventory, extract, and BS-Roformer-separate the 12 anime episodes.

NCOP/NCED are recorded but excluded from dialogue mining. The script is
restartable and never modifies radio-drama outputs or training lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import soundfile as sf


PKG = Path(
    r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
)
FFMPEG = PKG / "runtime" / "ffmpeg.exe"
FFPROBE = PKG / "runtime" / "ffprobe.exe"
BATCH = ROOT / "anime_batch_v1"
RAW = BATCH / "raw"
VOCALS = BATCH / "vocals"
OTHER = BATCH / "other"
TMP = BATCH / "tmp"
MANIFEST = BATCH / "manifest.json"
REPORT = BATCH / "report.json"

MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755"
EXPECTED_SR = 44100

EPISODES = [
    (
        f"majo_ep{index:02d}",
        Path(
            rf"F:/BaiduNetdiskDownload/[KTXP][Majo_no_Tabitabi][{index:02d}][BIG5][1080p][BDrip].mp4"
        ),
    )
    for index in range(1, 13)
]
EXCLUDED_ASSETS = [
    {
        "id": "majo_ncop",
        "source": str(
            Path(r"F:/BaiduNetdiskDownload/[KTXP][Majo_no_Tabitabi][NCOP][1080p][BDrip].mp4")
        ),
        "reason": "creditless opening is a song asset, not dialogue mining input",
    },
    {
        "id": "majo_nced",
        "source": str(
            Path(r"F:/BaiduNetdiskDownload/[KTXP][Majo_no_Tabitabi][NCED][1080p][BDrip].mp4")
        ),
        "reason": "creditless ending is a song asset, not dialogue mining input",
    },
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def fingerprint(path: Path) -> str:
    stat = path.stat()
    value = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def probe(path: Path) -> dict:
    command = [
        str(FFPROBE),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,channels,sample_rate:stream_tags=language,title",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    document = json.loads(result.stdout)
    audio_streams = [
        row for row in document.get("streams", []) if row.get("codec_type") == "audio"
    ]
    subtitle_streams = [
        row
        for row in document.get("streams", [])
        if row.get("codec_type") == "subtitle"
    ]
    if len(audio_streams) != 1:
        raise RuntimeError(f"Expected exactly one audio stream in {path}, got {len(audio_streams)}")
    audio = audio_streams[0]
    return {
        "duration_sec": float(document["format"]["duration"]),
        "container_size": int(document["format"]["size"]),
        "audio_stream_index": int(audio["index"]),
        "audio_codec": str(audio.get("codec_name") or ""),
        "audio_sample_rate": int(audio.get("sample_rate") or 0),
        "audio_channels": int(audio.get("channels") or 0),
        "audio_tags": audio.get("tags", {}),
        "subtitle_stream_count": len(subtitle_streams),
        "subtitle_streams": subtitle_streams,
    }


def valid_wave(path: Path, expected_duration: float) -> bool:
    if not path.is_file():
        return False
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return bool(
        info.samplerate == EXPECTED_SR
        and info.channels in {1, 2}
        and info.subtype == "PCM_16"
        and abs(float(info.duration) - expected_duration) <= 1.0
    )


def build_manifest() -> dict:
    missing = [str(path) for _, path in EPISODES if not path.is_file()]
    missing.extend(
        item["source"] for item in EXCLUDED_ASSETS if not Path(item["source"]).is_file()
    )
    if missing:
        raise FileNotFoundError("Missing anime assets: " + ", ".join(missing))
    tracks = []
    for index, (track_id, source) in enumerate(EPISODES, 1):
        metadata = probe(source)
        raw = RAW / f"{track_id}.wav"
        vocals = VOCALS / f"{track_id}_vocals.wav"
        other = OTHER / f"{track_id}_other.wav"
        tracks.append(
            {
                "id": track_id,
                "episode": index,
                "source": str(source.resolve()),
                "source_fingerprint": fingerprint(source),
                "raw": str(raw.resolve()),
                "vocals": str(vocals.resolve()),
                "other": str(other.resolve()),
                **metadata,
            }
        )
    excluded = []
    for item in EXCLUDED_ASSETS:
        source = Path(item["source"])
        excluded.append(
            {
                **item,
                "source": str(source.resolve()),
                "source_fingerprint": fingerprint(source),
                "metadata": probe(source),
            }
        )
    manifest = {
        "version": "anime_batch_v1_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "included": "12 main episodes only",
            "excluded": "NCOP/NCED song assets",
            "audio": "first and only 48kHz stereo AAC stream; subtitles are hardsubbed/no stream",
            "extraction": "44.1kHz stereo PCM16 WAV",
            "separation": MODEL_NAME,
            "training": "none; mining and human review only",
        },
        "tracks": tracks,
        "excluded_assets": excluded,
        "total_duration_sec": sum(row["duration_sec"] for row in tracks),
        "total_duration_min": sum(row["duration_sec"] for row in tracks) / 60.0,
    }
    write_json(MANIFEST, manifest)
    return manifest


def extract(manifest: dict, resume: bool) -> None:
    for index, row in enumerate(manifest["tracks"], 1):
        raw = Path(row["raw"])
        if resume and valid_wave(raw, float(row["duration_sec"])):
            print(f"[{index}/12] extract resume {row['id']}", flush=True)
            continue
        command = [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "2",
            "-i",
            str(row["source"]),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(EXPECTED_SR),
            "-c:a",
            "pcm_s16le",
            str(raw),
        ]
        print(f"[{index}/12] extract {row['id']}", flush=True)
        subprocess.run(command, check=True)
        if not valid_wave(raw, float(row["duration_sec"])):
            raise RuntimeError(f"Invalid extracted wave: {raw}")


def separate(manifest: dict, resume: bool) -> None:
    os.environ["TEMP"] = str(TMP)
    os.environ["TMP"] = str(TMP)
    os.environ["PATH"] = str(PKG / "runtime") + os.pathsep + os.environ.get("PATH", "")
    os.chdir(PKG)
    sys.path.insert(0, str(PKG / "tools" / "uvr5"))
    from bsroformer import Roformer_Loader

    print(f"load BS-Roformer {MODEL_NAME} on cuda:0 fp32", flush=True)
    loader = Roformer_Loader(
        model_path=str(PKG / "tools" / "uvr5" / "uvr5_weights" / f"{MODEL_NAME}.ckpt"),
        config_path=str(PKG / "tools" / "uvr5" / "uvr5_weights" / f"{MODEL_NAME}.yaml"),
        device="cuda:0",
        is_half=False,
    )
    for index, row in enumerate(manifest["tracks"], 1):
        raw = Path(row["raw"])
        vocals = Path(row["vocals"])
        other = Path(row["other"])
        if not valid_wave(raw, float(row["duration_sec"])):
            raise RuntimeError(f"Raw audio missing before separation: {raw}")
        if resume and valid_wave(vocals, float(row["duration_sec"])) and valid_wave(
            other, float(row["duration_sec"])
        ):
            print(f"[{index}/12] separation resume {row['id']}", flush=True)
            continue
        print(f"[{index}/12] separate {row['id']}", flush=True)
        loader._path_audio_(str(raw), str(OTHER), str(VOCALS), "wav", False)
        if not valid_wave(vocals, float(row["duration_sec"])) or not valid_wave(
            other, float(row["duration_sec"])
        ):
            raise RuntimeError(f"Separation outputs invalid for {row['id']}")


def summarize(manifest: dict, started: float) -> dict:
    tracks = []
    for row in manifest["tracks"]:
        item = {
            "id": row["id"],
            "episode": row["episode"],
            "duration_sec": row["duration_sec"],
            "raw_ready": valid_wave(Path(row["raw"]), float(row["duration_sec"])),
            "vocals_ready": valid_wave(Path(row["vocals"]), float(row["duration_sec"])),
            "other_ready": valid_wave(Path(row["other"]), float(row["duration_sec"])),
        }
        tracks.append(item)
    disk = shutil.disk_usage(BATCH.anchor)
    report = {
        "version": "anime_batch_v1_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "coverage": {
            "episodes": len(tracks),
            "raw_ready": sum(row["raw_ready"] for row in tracks),
            "vocals_ready": sum(row["vocals_ready"] for row in tracks),
            "other_ready": sum(row["other_ready"] for row in tracks),
            "duration_min": manifest["total_duration_min"],
        },
        "tracks": tracks,
        "disk_free_bytes": disk.free,
        "local_resources": RESOURCE_POLICY,
        "training_started": False,
        "existing_training_lists_modified": False,
        "score_90_claimed": False,
    }
    write_json(REPORT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("inventory", "extract", "separate", "all"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.time()
    for path in (BATCH, RAW, VOCALS, OTHER, TMP):
        path.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    if args.phase in {"extract", "all"}:
        extract(manifest, args.resume)
    if args.phase in {"separate", "all"}:
        separate(manifest, args.resume)
    report = summarize(manifest, started)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
