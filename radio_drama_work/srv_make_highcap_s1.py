#!/usr/bin/env python3
"""Create an isolated high-capacity D_old S1 refinement experiment on AutoDL."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

REPO = Path("/root/GPT-SoVITS")
SOURCE = Path("/root/autodl-tmp/exp/ireina_s2_old_v2")
NAME = os.environ.get("S1_EXP", "ireina_s1_dold_hicap_b8s28_v1")
EXP = Path("/root/autodl-tmp/exp") / NAME
OUT = Path("/root/autodl-tmp/weights") / ("GPT_" + NAME)

EPOCHS = int(os.environ.get("S1_EPOCHS", "2"))
BATCH = int(os.environ.get("S1_BATCH", "8"))
MAX_SEC = int(os.environ.get("S1_MAX_SEC", "28"))
SAVE_EVERY = int(os.environ.get("S1_SAVE_EVERY", "1"))

if EXP.exists():
    raise SystemExit(f"experiment already exists: {EXP}")

for path in (EXP, OUT, EXP / "cfg", EXP / "logs_s1_v2Pro"):
    path.mkdir(parents=True, exist_ok=False)

for name in ("2-name2text.txt", "6-name2semantic.tsv"):
    shutil.copy2(SOURCE / name, EXP / name)

for name in ("3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn"):
    (EXP / name).symlink_to(SOURCE / name, target_is_directory=True)

with (REPO / "GPT_SoVITS/configs/s1longer-v2.yaml").open(encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

cfg["train"].update(
    {
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "save_every_n_epoch": SAVE_EVERY,
        "if_save_latest": False,
        "if_save_every_weights": True,
        "half_weights_save_dir": str(OUT),
        "exp_name": NAME,
        "gpu_numbers": "0",
        "precision": "16-mixed",
    }
)
cfg["data"].update({"max_sec": MAX_SEC, "num_workers": 4})
cfg["train_semantic_path"] = str(EXP / "6-name2semantic.tsv")
cfg["train_phoneme_path"] = str(EXP / "2-name2text.txt")
cfg["output_dir"] = str(EXP / "logs_s1_v2Pro")
cfg["pretrained_s1"] = str(REPO / "GPT_SoVITS/pretrained_models/s1v3.ckpt")

with (EXP / "cfg/s1.yaml").open("w", encoding="utf-8") as fp:
    yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)

print(f"created {EXP}")
print(f"epochs={EPOCHS} batch={BATCH} max_sec={MAX_SEC} save_every={SAVE_EVERY}")
print(f"weights={OUT}")
