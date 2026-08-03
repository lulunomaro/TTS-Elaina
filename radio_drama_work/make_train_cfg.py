# -*- coding: utf-8 -*-
"""生成合并数据集的 S1/S2 训练配置，并预建所有 checkpoint 目录（Windows 下必须预建，否则保存时 FileNotFoundError）。"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
EXP = "my_voice_merged"
OPT = "logs/%s" % EXP
TMP = os.path.join(ROOT, "logs", EXP, "tmp")
os.makedirs(TMP, exist_ok=True)

# ---------- S1 (GPT) ----------
s1 = {
    "data": {"max_eval_sample": 8, "max_sec": 20, "num_workers": 2, "pad_val": 1024},
    "inference": {"top_k": 15},
    "model": {
        "EOS": 1024,
        "dropout": 0,
        "embedding_dim": 512,
        "head": 16,
        "hidden_dim": 512,
        "linear_units": 2048,
        "n_layer": 24,
        "phoneme_vocab_size": 732,
        "random_bert": 0,
        "vocab_size": 1025,
    },
    "optimizer": {"decay_steps": 40000, "lr": 0.01, "lr_end": 0.0001, "lr_init": 1.0e-05, "warmup_steps": 2000},
    "output_dir": "%s/logs_s1_v2Pro" % OPT,
    "pretrained_s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "train": {
        "batch_size": 4,
        "epochs": 40,
        "exp_name": EXP,
        "gradient_clip": 1.0,
        "half_weights_save_dir": "GPT_weights_v2Pro_merged",
        "if_dpo": 0,
        "if_save_every_weights": 1,
        "if_save_latest": 1,
        "precision": "16-mixed",
        "save_every_n_epoch": 4,
        "seed": 1234,
    },
    "train_phoneme_path": "%s/2-name2text.txt" % OPT,
    "train_semantic_path": "%s/6-name2semantic.tsv" % OPT,
}

import yaml  # noqa: E402

p1 = os.path.join(TMP, "s1_merged.yaml")
with open(p1, "w", encoding="utf-8") as f:
    yaml.safe_dump(s1, f, allow_unicode=True, sort_keys=True)

# ---------- S2 (SoVITS) ----------
s2 = json.load(open(os.path.join(ROOT, "logs", "my_voice", "tmp", "local_s2_full.json"), encoding="utf-8"))
s2["train"]["batch_size"] = 3
s2["train"]["epochs"] = 40
s2["train"]["save_every_epoch"] = 4
s2["data"]["exp_dir"] = OPT
s2["s2_ckpt_dir"] = OPT
s2["save_weight_dir"] = "SoVITS_weights_v2Pro_merged"
s2["name"] = EXP
p2 = os.path.join(TMP, "s2_merged.json")
with open(p2, "w", encoding="utf-8") as f:
    json.dump(s2, f, ensure_ascii=False, indent=1)

# ---------- 预建目录 ----------
dirs = [
    os.path.join(ROOT, "GPT_weights_v2Pro_merged"),
    os.path.join(ROOT, "SoVITS_weights_v2Pro_merged"),
    os.path.join(ROOT, OPT.replace("/", os.sep), "logs_s1_v2Pro"),
    os.path.join(ROOT, OPT.replace("/", os.sep), "logs_s2_v2Pro"),
    os.path.join(ROOT, OPT.replace("/", os.sep), "eval"),
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

print("[s1 cfg]", p1)
print("[s2 cfg]", p2)
print("[dirs] 已预建:")
for d in dirs:
    print("   ", d)
