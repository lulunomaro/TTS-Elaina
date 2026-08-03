#!/usr/bin/env python3
"""Create an isolated low-LR SoVITS fine-tune experiment."""
import json
import os
import shutil
import sys

BASE = "/root/autodl-tmp/exp/ireina_s2_emo_focus_v2_b16_lr75e5_dlr1e5"
EXP = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/exp/ireina_s2_b16e2_micro_v1"
G = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/weights/ireina_s2_b16e2_warmstart.pth"
OUT_WEIGHTS = f"/root/autodl-tmp/weights/SoVITS_{os.path.basename(EXP)}"
CFG_DIR = os.path.join(EXP, "cfg")

if os.path.exists(EXP):
    raise SystemExit(f"experiment already exists: {EXP}")
os.makedirs(CFG_DIR, exist_ok=False)

for name in ("2-name2text.txt", "6-name2semantic.tsv"):
    shutil.copy2(os.path.join(BASE, name), os.path.join(EXP, name))
for name in ("3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn"):
    os.symlink(os.path.join(BASE, name), os.path.join(EXP, name), target_is_directory=True)

with open(os.path.join(BASE, "cfg", "s2.json"), encoding="utf-8") as fp:
    cfg = json.load(fp)
train = cfg["train"]
train.update({
    "epochs": 2,
    "batch_size": 24,
    "save_every_epoch": 1,
    "if_save_latest": False,
    "if_save_every_weights": True,
    "learning_rate": 3.0e-5,
    "d_learning_rate": 7.5e-6,
    "text_low_lr_rate": 0.10,
    "pretrained_s2G": G,
    "pretrained_s2D": "/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
    "gpu_numbers": "0",
    "grad_ckpt": False,
    "fp16_run": True,
})
cfg["data"]["exp_dir"] = EXP
cfg["s2_ckpt_dir"] = EXP
cfg["save_weight_dir"] = OUT_WEIGHTS
cfg["name"] = os.path.basename(EXP)
cfg["version"] = "v2Pro"
cfg["model"]["version"] = "v2Pro"
os.makedirs(OUT_WEIGHTS, exist_ok=True)
with open(os.path.join(CFG_DIR, "s2.json"), "w", encoding="utf-8") as fp:
    json.dump(cfg, fp, indent=2, ensure_ascii=False)
print(f"EXP={EXP}")
print(f"CFG={CFG_DIR}/s2.json")
print(f"WEIGHTS={OUT_WEIGHTS}")
print(f"G={G}")
print(f"batch={train['batch_size']} epochs={train['epochs']} lr={train['learning_rate']} d_lr={train['d_learning_rate']}")
