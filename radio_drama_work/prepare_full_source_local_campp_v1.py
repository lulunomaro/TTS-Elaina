# -*- coding: utf-8 -*-
"""Prepare source-local CAM++ speech-window caches for all 12 radio dramas.

This is a resumable cache-building step.  It reuses the seven already valid
FunASR/CAM++ caches and computes only the five previously uncovered sources.
Heavy speaker encoding runs on CUDA; CPU work is capped by the shared local
resource policy so CAD work can remain responsive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_resource_policy import apply_cpu_friendly_policy


RESOURCE_POLICY = apply_cpu_friendly_policy(2)

import numpy as np
import torch
from funasr import AutoModel

import full_funasr_target_recall_v1 as recall
import full_speaker_diarize_new_v1 as base


MANIFEST = ROOT / "full_track_manifest_v1.json"
LEGACY = ROOT / "full_funasr_target_recall_v2"
OUT = ROOT / "full_source_local_campp_v1"
EMBED_CACHE = OUT / "embedding_cache"
VAD_CACHE = OUT / "vad_cache"
REPORT = OUT / "report.json"
TARGET_REFS = EMBED_CACHE / "all_target_references.npz"
VAD_MODEL = recall.VAD_PATH


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )


def source_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{recall.WINDOW_SEC}|{recall.HOP_SEC}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def cache_is_valid(path: Path, duration: float) -> tuple[bool, str, int]:
    if not path.is_file():
        return False, "missing", 0
    try:
        data = np.load(str(path), allow_pickle=True)
        specs = np.asarray(data["specs"], dtype=np.float32)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"read_error:{exc!r}", 0
    if specs.ndim != 2 or specs.shape[1] != 2:
        return False, "invalid_specs_shape", 0
    if embeddings.ndim != 2 or embeddings.shape[0] != specs.shape[0]:
        return False, "invalid_embedding_shape", 0
    if embeddings.shape[1] != 192:
        return False, "unexpected_embedding_dimension", 0
    if not len(specs):
        return False, "empty", 0
    if not np.all(np.isfinite(specs)) or not np.all(np.isfinite(embeddings)):
        return False, "non_finite", 0
    if float(specs[:, 0].min()) < -0.01 or float(specs[:, 1].max()) > duration + 0.10:
        return False, "time_out_of_range", 0
    return True, "ok", int(len(specs))


def load_vad_rows(vad: AutoModel, source: Path) -> tuple[list[list[int]], str]:
    local = VAD_CACHE / f"{source.stem}.json"
    legacy = LEGACY / "vad_cache" / f"{source.stem}.json"
    for candidate, method in ((local, "resume"), (legacy, "legacy_reuse")):
        if not candidate.is_file():
            continue
        try:
            rows = json.loads(candidate.read_text(encoding="utf-8"))["value"]
            if rows:
                if candidate != local:
                    shutil.copyfile(candidate, local)
                return rows, method
        except Exception:
            pass
    generated = vad.generate(input=str(source))[0]
    rows = generated.get("value", [])
    if not rows:
        raise RuntimeError(f"FunASR VAD returned no speech for {source}")
    write_json(local, {"source": str(source), "value": rows})
    return rows, "gpu_generated"


def build_embeddings(
    cam: AutoModel, vad: AutoModel, source: Path
) -> tuple[int, int, str]:
    audio, sample_rate = base.read_mono(source)
    duration = len(audio) / sample_rate
    audio16 = base.resample_1d(audio, sample_rate, recall.CAM_SR)
    vad_rows, vad_method = load_vad_rows(vad, source)

    chunks = recall.make_chunks(vad_rows, audio16)
    if not chunks:
        raise RuntimeError(f"No CAM++ windows built for {source}")
    embeddings = base.cam_embed_batch(cam, [row[2] for row in chunks])
    specs = np.asarray([[row[0], row[1]] for row in chunks], dtype=np.float32)
    destination = EMBED_CACHE / f"{source.stem}.npz"
    np.savez_compressed(
        str(destination),
        specs=specs,
        embeddings=embeddings.numpy().astype(np.float32),
    )
    return len(vad_rows), len(chunks), vad_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    VAD_CACHE.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tracks = manifest["tracks"]
    if len(tracks) != 12:
        raise RuntimeError(f"Expected 12 tracks, found {len(tracks)}")

    legacy_refs = LEGACY / "embedding_cache" / "all_target_references.npz"
    if not legacy_refs.is_file():
        raise FileNotFoundError(legacy_refs)
    shutil.copyfile(legacy_refs, TARGET_REFS)

    rows: list[dict] = []
    pending: list[tuple[dict, Path, float]] = []
    for track in tracks:
        source = Path(track["vocals"])
        if not source.is_file():
            raise FileNotFoundError(source)
        audio_info = base.sf.info(str(source))
        duration = float(audio_info.duration)
        destination = EMBED_CACHE / f"{source.stem}.npz"
        valid, reason, count = cache_is_valid(destination, duration)
        if args.resume and valid:
            rows.append(
                {
                    "source_id": track["id"],
                    "source": str(source),
                    "fingerprint": source_fingerprint(source),
                    "duration_sec": duration,
                    "cache": str(destination),
                    "chunk_count": count,
                    "method": "resume",
                }
            )
            print(f"[{track['id']}] resume chunks={count}", flush=True)
            continue

        legacy = LEGACY / "embedding_cache" / f"{source.stem}.npz"
        legacy_valid, _, legacy_count = cache_is_valid(legacy, duration)
        if legacy_valid:
            shutil.copyfile(legacy, destination)
            rows.append(
                {
                    "source_id": track["id"],
                    "source": str(source),
                    "fingerprint": source_fingerprint(source),
                    "duration_sec": duration,
                    "cache": str(destination),
                    "chunk_count": legacy_count,
                    "method": "legacy_reuse",
                }
            )
            print(f"[{track['id']}] legacy reuse chunks={legacy_count}", flush=True)
        else:
            pending.append((track, source, duration))
            print(f"[{track['id']}] pending GPU encoding ({reason})", flush=True)

    cam = None
    vad = None
    if pending:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[model] CAM++ device={device}; pending={len(pending)}", flush=True)
        cam = AutoModel(
            model=str(base.CAM_MODEL),
            device=device,
            check_latest=False,
            disable_pbar=True,
        )
        vad = AutoModel(
            model=str(VAD_MODEL),
            device=device,
            disable_update=True,
            disable_pbar=True,
        )

    for track, source, duration in pending:
        source_started = time.time()
        vad_count, chunk_count, vad_method = build_embeddings(cam, vad, source)
        destination = EMBED_CACHE / f"{source.stem}.npz"
        valid, reason, verified_count = cache_is_valid(destination, duration)
        if not valid or verified_count != chunk_count:
            raise RuntimeError(f"Invalid generated cache for {source}: {reason}")
        row = {
            "source_id": track["id"],
            "source": str(source),
            "fingerprint": source_fingerprint(source),
            "duration_sec": duration,
            "cache": str(destination),
            "chunk_count": chunk_count,
            "vad_segment_count": vad_count,
            "vad_method": vad_method,
            "method": "gpu_generated",
            "elapsed_sec": time.time() - source_started,
        }
        rows.append(row)
        print(
            f"[{track['id']}] generated vad={vad_count} chunks={chunk_count} "
            f"elapsed={row['elapsed_sec']:.1f}s",
            flush=True,
        )
        partial = {
            "version": "full_source_local_campp_v1_partial",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sources": sorted(rows, key=lambda value: value["source_id"]),
        }
        write_json(REPORT, partial)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cam is not None:
        del cam
    if vad is not None:
        del vad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows.sort(key=lambda value: value["source_id"])
    ids = [row["source_id"] for row in rows]
    if len(rows) != 12 or len(ids) != len(set(ids)):
        raise RuntimeError(f"Cache coverage failed: {len(rows)} rows / {len(set(ids))} unique")
    report = {
        "version": "full_source_local_campp_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - started,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "local_resources": RESOURCE_POLICY,
        "parameters": {
            "window_sec": recall.WINDOW_SEC,
            "hop_sec": recall.HOP_SEC,
            "embedding_dimension": 192,
            "vad_model": str(VAD_MODEL),
            "speaker_model": str(base.CAM_MODEL),
        },
        "coverage": {"expected": 12, "completed": len(rows)},
        "method_counts": {
            method: sum(row["method"] == method for row in rows)
            for method in sorted({row["method"] for row in rows})
        },
        "total_duration_sec": sum(row["duration_sec"] for row in rows),
        "total_chunk_count": sum(row["chunk_count"] for row in rows),
        "target_reference_cache": str(TARGET_REFS),
        "sources": rows,
    }
    write_json(REPORT, report)
    print(json.dumps(report["coverage"] | report["method_counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
