from __future__ import annotations

import os
from pathlib import Path

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HOME", r"radio_drama_work/hf_cache")

from huggingface_hub import hf_hub_download

OUT = Path(r"radio_drama_work/clearvoice_sr_smoke_v1/checkpoints/MossFormer2_SR_48K")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for filename in ("last_best_checkpoint_g.pt",):
        print(f"downloading={filename}", flush=True)
        path = hf_hub_download(
            repo_id="alibabasglab/MossFormer2_SR_48K",
            filename=filename,
            local_dir=str(OUT),
            force_download=False,
            resume_download=True,
        )
        print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
