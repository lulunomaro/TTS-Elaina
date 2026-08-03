"""Resumable range downloader for the public AudioSR speech checkpoint."""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(r"radio_drama_work/radio_drama_work")
CACHE = ROOT / "hf_cache" / "hub" / "models--haoheliu--audiosr_speech" / "blobs"
NAME = "59fd60a9b93346c698788bac9b6077977a21733e7a78db9d3c3af3f1bdebc640.incomplete"
PART = CACHE / NAME
FINAL = CACHE / NAME.removesuffix(".incomplete")
RESOLVE = "https://huggingface.co/haoheliu/audiosr_speech/resolve/main/pytorch_model.bin"
EXPECTED_SHA256 = NAME.removesuffix(".incomplete")
PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
CHUNK = 32 * 1024 * 1024


def get_total(session: requests.Session) -> tuple[int, str]:
    response = session.get(RESOLVE, headers={"Range": "bytes=0-0"}, allow_redirects=True, stream=True, timeout=(30, 60))
    response.raise_for_status()
    match = re.search(r"bytes/s+0-0/(/d+)", response.headers.get("content-range", ""))
    if not match:
        raise RuntimeError(f"missing content-range: {response.status_code} {dict(response.headers)}")
    return int(match.group(1)), response.url


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.proxies.update(PROXIES)
    total, _ = get_total(session)
    current = PART.stat().st_size if PART.exists() else 0
    if current > total:
        raise RuntimeError(f"partial file is larger than remote: {current} > {total}")
    print(f"total={total} current={current} remaining={total-current}", flush=True)
    while current < total:
        end = min(total - 1, current + CHUNK - 1)
        last_error = None
        for attempt in range(1, 9):
            try:
                response = session.get(RESOLVE, headers={"Range": f"bytes={current}-{end}"}, allow_redirects=True, stream=True, timeout=(30, 180))
                response.raise_for_status()
                content_range = response.headers.get("content-range", "")
                match = re.search(r"bytes/s+(/d+)-(/d+)/(/d+)", content_range)
                if not match or int(match.group(1)) != current or int(match.group(2)) != end or int(match.group(3)) != total:
                    raise RuntimeError(f"bad range response: {response.status_code} {content_range}")
                expected = end - current + 1
                received = 0
                with PART.open("ab") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
                            received += len(block)
                if received != expected:
                    raise RuntimeError(f"short chunk: {received} != {expected}")
                current = end + 1
                print(f"downloaded={current}/{total} ({current/total:.1%})", flush=True)
                break
            except Exception as exc:  # network/CDN retries are expected here
                last_error = exc
                print(f"retry chunk={current}-{end} attempt={attempt}: {exc}", flush=True)
                time.sleep(min(10, attempt * 2))
        else:
            raise RuntimeError(f"chunk failed after retries: {last_error}")
    print("computing_sha256", flush=True)
    digest = hashlib.sha256()
    with PART.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    print(f"sha256={actual}", flush=True)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"sha256 mismatch: {actual} != {EXPECTED_SHA256}")
    if FINAL.exists():
        FINAL.unlink()
    PART.replace(FINAL)
    print(f"complete={FINAL}", flush=True)


if __name__ == "__main__":
    download()
