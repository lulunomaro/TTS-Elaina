#!/usr/bin/env python3
"""Extend an existing high-capacity S1 run without changing its data."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

name = os.environ.get("S1_EXP", "ireina_s1_dold_hicap_b8s28_v1")
epochs = int(os.environ.get("S1_EPOCHS", "12"))
save_every = int(os.environ.get("S1_SAVE_EVERY", "2"))
path = Path("/root/autodl-tmp/exp") / name / "cfg/s1.yaml"
if not path.exists():
    raise SystemExit(f"missing config: {path}")
with path.open(encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)
cfg["train"]["epochs"] = epochs
cfg["train"]["save_every_n_epoch"] = save_every
cfg["train"]["if_save_latest"] = False
cfg["train"]["if_save_every_weights"] = True
with path.open("w", encoding="utf-8") as fp:
    yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)
print(f"extended {path}: epochs={epochs}, save_every={save_every}")
