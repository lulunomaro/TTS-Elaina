# -*- coding: utf-8 -*-
"""在 AutoDL 服务器上生成 v2Pro 的 S1/S2 训练配置。
所有产物写入 /root/autodl-tmp（数据盘），避免撑爆仅剩 4.5G 的系统盘。
"""
import json
import os
import sys

import yaml

REPO = "/root/GPT-SoVITS"
EXP_NAME = sys.argv[1] if len(sys.argv) > 1 else "ireina_v1"
EXP_DIR = "/root/autodl-tmp/exp/%s" % EXP_NAME
CFG_DIR = os.path.join(EXP_DIR, "cfg")

# ---- 官方推荐轮数（关键：上一轮本地训练 40 轮严重过拟合，是机械感主因）----
S1_EPOCHS = int(os.environ.get("S1_EPOCHS", 15))   # webui 默认 15
S1_BATCH = int(os.environ.get("S1_BATCH", 6))
S1_SAVE_EVERY = int(os.environ.get("S1_SAVE_EVERY", 3))
# max_sec=54(官方) + batch8 在 5090 32G 上第 4 轮 OOM；本地统计 22s 已覆盖 96%+ 数据
S1_MAX_SEC = int(os.environ.get("S1_MAX_SEC", 22))

S2_EPOCHS = int(os.environ.get("S2_EPOCHS", 8))    # webui v2 系默认 8
S2_BATCH = int(os.environ.get("S2_BATCH", 8))      # 本地只能 3
S2_SAVE_EVERY = int(os.environ.get("S2_SAVE_EVERY", 2))

GPT_SAVE_DIR = "/root/autodl-tmp/weights/GPT_weights_v2Pro"
SOVITS_SAVE_DIR = "/root/autodl-tmp/weights/SoVITS_weights_v2Pro"

for d in (CFG_DIR, GPT_SAVE_DIR, SOVITS_SAVE_DIR,
          os.path.join(EXP_DIR, "logs_s1_v2Pro"),
          os.path.join(EXP_DIR, "logs_s2_v2Pro"),
          os.path.join(EXP_DIR, "eval")):
    os.makedirs(d, exist_ok=True)

# ---------------- S1 (GPT) ----------------
with open(os.path.join(REPO, "GPT_SoVITS/configs/s1longer-v2.yaml"), "r", encoding="utf-8") as f:
    s1 = yaml.safe_load(f)

s1["train"]["epochs"] = S1_EPOCHS
s1["train"]["batch_size"] = S1_BATCH
s1["train"]["save_every_n_epoch"] = S1_SAVE_EVERY
s1["train"]["if_save_latest"] = False        # 保留每个中间 ckpt，便于挑最优
s1["train"]["if_save_every_weights"] = True
s1["train"]["half_weights_save_dir"] = GPT_SAVE_DIR
s1["train"]["exp_name"] = EXP_NAME
s1["train"]["gpu_numbers"] = "0"
s1["train"]["precision"] = "16-mixed"
s1["train_semantic_path"] = "%s/6-name2semantic.tsv" % EXP_DIR
s1["train_phoneme_path"] = "%s/2-name2text.txt" % EXP_DIR
s1["output_dir"] = "%s/logs_s1_v2Pro" % EXP_DIR
s1["pretrained_s1"] = "%s/GPT_SoVITS/pretrained_models/s1v3.ckpt" % REPO
s1["data"]["max_sec"] = S1_MAX_SEC

s1_path = os.path.join(CFG_DIR, "s1.yaml")
with open(s1_path, "w", encoding="utf-8") as f:
    yaml.dump(s1, f, allow_unicode=True, sort_keys=False)

# ---------------- S2 (SoVITS) ----------------
with open(os.path.join(REPO, "GPT_SoVITS/configs/s2v2Pro.json"), "r", encoding="utf-8") as f:
    s2 = json.load(f)

s2["train"]["batch_size"] = S2_BATCH
s2["train"]["epochs"] = S2_EPOCHS
s2["train"]["save_every_epoch"] = S2_SAVE_EVERY
s2["train"]["if_save_latest"] = False
s2["train"]["if_save_every_weights"] = True
s2["train"]["text_low_lr_rate"] = 0.4
s2["train"]["pretrained_s2G"] = "%s/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth" % REPO
s2["train"]["pretrained_s2D"] = "%s/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth" % REPO
s2["train"]["gpu_numbers"] = "0"
s2["train"]["grad_ckpt"] = False
s2["train"]["fp16_run"] = True
s2["train"]["lora_rank"] = 32
s2["model"]["version"] = "v2Pro"
s2["data"]["exp_dir"] = EXP_DIR
s2["s2_ckpt_dir"] = EXP_DIR
s2["save_weight_dir"] = SOVITS_SAVE_DIR
s2["name"] = EXP_NAME
s2["version"] = "v2Pro"

s2_path = os.path.join(CFG_DIR, "s2.json")
with open(s2_path, "w", encoding="utf-8") as f:
    json.dump(s2, f, indent=2, ensure_ascii=False)

print("EXP_DIR      : %s" % EXP_DIR)
print("S1 config    : %s   (epochs=%d batch=%d save_every=%d max_sec=%d)"
      % (s1_path, S1_EPOCHS, S1_BATCH, S1_SAVE_EVERY, S1_MAX_SEC))
print("S2 config    : %s   (epochs=%d batch=%d save_every=%d)"
      % (s2_path, S2_EPOCHS, S2_BATCH, S2_SAVE_EVERY))
print("GPT weights  : %s" % GPT_SAVE_DIR)
print("SoVITS weights: %s" % SOVITS_SAVE_DIR)
