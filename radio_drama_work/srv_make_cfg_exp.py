# -*- coding: utf-8 -*-
"""为独立 v2Pro 实验生成 S1/S2 配置，所有输出均按实验名隔离。"""

import json
import os
import sys

import yaml


if len(sys.argv) < 2:
    raise SystemExit("usage: srv_make_cfg_exp.py EXP_NAME")

REPO = "/root/GPT-SoVITS"
EXP_NAME = sys.argv[1]
EXP_DIR = f"/root/autodl-tmp/exp/{EXP_NAME}"
CFG_DIR = f"{EXP_DIR}/cfg"
GPT_SAVE_DIR = f"/root/autodl-tmp/weights/GPT_{EXP_NAME}"
SOVITS_SAVE_DIR = f"/root/autodl-tmp/weights/SoVITS_{EXP_NAME}"

S1_EPOCHS = int(os.environ.get("S1_EPOCHS", "9"))
S1_BATCH = int(os.environ.get("S1_BATCH", "6"))
S1_SAVE_EVERY = int(os.environ.get("S1_SAVE_EVERY", "3"))
S1_MAX_SEC = int(os.environ.get("S1_MAX_SEC", "22"))
S2_EPOCHS = int(os.environ.get("S2_EPOCHS", "8"))
S2_BATCH = int(os.environ.get("S2_BATCH", "8"))
S2_SAVE_EVERY = int(os.environ.get("S2_SAVE_EVERY", "2"))
S2_LR = float(os.environ.get("S2_LR", "0.0001"))
S2_TEXT_LR = float(os.environ.get("S2_TEXT_LR", "0.4"))
S2_D_LR = float(os.environ.get("S2_D_LR", str(S2_LR)))
S2_PRETRAIN_G = os.environ.get("S2_PRETRAIN_G", f"{REPO}/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth")
S2_PRETRAIN_D = os.environ.get("S2_PRETRAIN_D", f"{REPO}/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth")

for path in (
    CFG_DIR,
    GPT_SAVE_DIR,
    SOVITS_SAVE_DIR,
    f"{EXP_DIR}/logs_s1_v2Pro",
    f"{EXP_DIR}/logs_s2_v2Pro",
    f"{EXP_DIR}/eval",
):
    os.makedirs(path, exist_ok=True)

with open(f"{REPO}/GPT_SoVITS/configs/s1longer-v2.yaml", encoding="utf-8") as fp:
    s1 = yaml.safe_load(fp)
s1["train"].update(
    {
        "epochs": S1_EPOCHS,
        "batch_size": S1_BATCH,
        "save_every_n_epoch": S1_SAVE_EVERY,
        "if_save_latest": False,
        "if_save_every_weights": True,
        "half_weights_save_dir": GPT_SAVE_DIR,
        "exp_name": EXP_NAME,
        "gpu_numbers": "0",
        "precision": "16-mixed",
    }
)
s1["train_semantic_path"] = f"{EXP_DIR}/6-name2semantic.tsv"
s1["train_phoneme_path"] = f"{EXP_DIR}/2-name2text.txt"
s1["output_dir"] = f"{EXP_DIR}/logs_s1_v2Pro"
s1["pretrained_s1"] = f"{REPO}/GPT_SoVITS/pretrained_models/s1v3.ckpt"
s1["data"]["max_sec"] = S1_MAX_SEC
with open(f"{CFG_DIR}/s1.yaml", "w", encoding="utf-8") as fp:
    yaml.dump(s1, fp, allow_unicode=True, sort_keys=False)

with open(f"{REPO}/GPT_SoVITS/configs/s2v2Pro.json", encoding="utf-8") as fp:
    s2 = json.load(fp)
s2["train"].update(
    {
        "batch_size": S2_BATCH,
        "epochs": S2_EPOCHS,
        "save_every_epoch": S2_SAVE_EVERY,
        "if_save_latest": False,
        "if_save_every_weights": True,
        "learning_rate": S2_LR,
        "d_learning_rate": S2_D_LR,
        "text_low_lr_rate": S2_TEXT_LR,
        "pretrained_s2G": S2_PRETRAIN_G,
        "pretrained_s2D": S2_PRETRAIN_D,
        "gpu_numbers": "0",
        "grad_ckpt": False,
        "fp16_run": True,
    }
)
s2["data"]["exp_dir"] = EXP_DIR
s2["s2_ckpt_dir"] = EXP_DIR
s2["save_weight_dir"] = SOVITS_SAVE_DIR
s2["name"] = EXP_NAME
s2["version"] = "v2Pro"
s2["model"]["version"] = "v2Pro"
with open(f"{CFG_DIR}/s2.json", "w", encoding="utf-8") as fp:
    json.dump(s2, fp, indent=2, ensure_ascii=False)

print(f"EXP_DIR={EXP_DIR}")
print(f"S1 epochs={S1_EPOCHS} batch={S1_BATCH} max_sec={S1_MAX_SEC}")
print(f"S2 epochs={S2_EPOCHS} batch={S2_BATCH} save_every={S2_SAVE_EVERY}")
print(f"S2 learning_rate={S2_LR} text_low_lr_rate={S2_TEXT_LR}")
print(f"S2 d_learning_rate={S2_D_LR}")
print(f"S2 pretrained_s2G={S2_PRETRAIN_G}")
print(f"S2 pretrained_s2D={S2_PRETRAIN_D}")
print(f"GPT_SAVE_DIR={GPT_SAVE_DIR}")
print(f"SOVITS_SAVE_DIR={SOVITS_SAVE_DIR}")
