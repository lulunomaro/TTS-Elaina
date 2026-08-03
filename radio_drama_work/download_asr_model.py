# -*- coding: utf-8 -*-
"""通过本机系统代理(127.0.0.1:7897)从 HuggingFace 下载 faster-whisper large-v3-turbo。
落盘位置与 tools/asr/fasterwhisper_asr.py 的 download_model() 约定一致。"""
import os
import time

PROXY = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.pop("HF_ENDPOINT", None)  # 走官方端点

PKG = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
LOCAL = os.path.join(PKG, "tools", "asr", "models", "faster-whisper-large-v3-turbo")
REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
FILES = ["config.json", "model.bin", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"]

from huggingface_hub import snapshot_download  # noqa: E402

t0 = time.time()
print(f"[download] repo={REPO}")
print(f"[download] dest ={LOCAL}")
print(f"[download] proxy={PROXY}")
os.makedirs(LOCAL, exist_ok=True)

path = snapshot_download(
    repo_id=REPO,
    local_dir=LOCAL,
    local_dir_use_symlinks=False,
    allow_patterns=FILES,
    max_workers=4,
)
print(f"[download] OK -> {path}  ({time.time() - t0:.1f}s)")

for f in FILES:
    p = os.path.join(LOCAL, f)
    size = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {'OK ' if size else 'MISSING'} {f}  {size / 1024 / 1024:.1f} MB")
