#!/usr/bin/env python3
"""Create a text/韵律-only refinement from the b16 e2 warm-start."""
import json
import os
import shutil

BASE = "/root/autodl-tmp/exp/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5"
EXP = os.environ.get("TEXT_EXP", "/root/autodl-tmp/exp/ireina_s2_b16e2_textonly_v1")
G = os.environ.get("TEXT_G", "/root/autodl-tmp/weights/ireina_s2_b16e2_warmstart.pth")
OUT = os.environ.get("TEXT_OUT", "/root/autodl-tmp/weights/SoVITS_ireina_s2_b16e2_textonly_v1")
LR = float(os.environ.get("TEXT_LR", "1.0e-5"))
NAME = os.path.basename(EXP)

if os.path.exists(EXP):
    raise SystemExit(f"experiment already exists: {EXP}")
os.makedirs(os.path.join(EXP, "cfg"), exist_ok=False)
for name in ("2-name2text.txt", "6-name2semantic.tsv"):
    shutil.copy2(os.path.join(BASE, name), os.path.join(EXP, name))
for name in ("3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn"):
    os.symlink(os.path.join(BASE, name), os.path.join(EXP, name), target_is_directory=True)

with open(os.path.join(BASE, "cfg", "s2.json"), encoding="utf-8") as fp:
    cfg = json.load(fp)
cfg["train"].update({
    "epochs": 1,
    "batch_size": 24,
    "save_every_epoch": 1,
    "if_save_latest": False,
    "if_save_every_weights": True,
    "learning_rate": LR,
    "d_learning_rate": 0.0,
    "text_low_lr_rate": 1.0,
    "text_only": True,
    "max_steps_per_epoch": 17,
    "pretrained_s2G": G,
    "pretrained_s2D": "/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
    "gpu_numbers": "0",
    "grad_ckpt": False,
    "fp16_run": True,
})
cfg["data"]["exp_dir"] = EXP
cfg["s2_ckpt_dir"] = EXP
cfg["save_weight_dir"] = OUT
cfg["name"] = NAME
cfg["version"] = "v2Pro"
cfg["model"]["version"] = "v2Pro"
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(EXP, "cfg", "s2.json"), "w", encoding="utf-8") as fp:
    json.dump(cfg, fp, indent=2, ensure_ascii=False)
print(f"created {EXP} batch=24 epochs=1 text_only lr={LR}")
