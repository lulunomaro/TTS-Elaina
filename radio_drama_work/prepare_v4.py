# -*- coding: utf-8 -*-
"""在 V4 包内为同一份合并数据集做预处理（用于训练 V4 SoVITS LoRA）。

V4 与 v2Pro 的差异：
  - gin_channels 512（configs/s2.json），语义提取须用 512 的 G（s2G488k.pth）
  - SoVITS LoRA 只需 2-name2text.txt / 4-cnhubert / 5-wav32k，不需要 7-sv_cn
  - GPT(S1) 与 v2Pro 同源（s1v3.ckpt 字节一致），直接复用 v2Pro 训好的权重
"""
import io
import os
import shutil
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V2 = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
V4 = r"radio_drama_work/GPT-SoVITS-v4-20250529-nvidia50/GPT-SoVITS-v4-20250529-nvidia50"
PY = os.path.join(V4, "runtime", "python.exe")
PD = os.path.join(V4, "GPT_SoVITS", "prepare_datasets")
EXP = "my_voice_merged"
OPT = os.path.join(V4, "logs", EXP)
PM = os.path.join(V4, "GPT_SoVITS", "pretrained_models")

os.makedirs(OPT, exist_ok=True)

# 复用同一份 merged.list（内部是绝对路径，跨包可用）
SRC_LIST = os.path.join(V2, "logs", EXP, "merged.list")
LIST = os.path.join(OPT, "merged.list")
shutil.copy2(SRC_LIST, LIST)
print("[list] 复制 merged.list ->", LIST)

base = dict(os.environ)
base.update(
    {
        "inp_text": LIST,
        "inp_wav_dir": "",
        "exp_name": EXP,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "opt_dir": OPT,
        "is_half": "True",
        "version": "v4",
        "PYTHONIOENCODING": "utf-8",
    }
)

steps = [
    ("1-get-text.py", {"bert_pretrained_dir": os.path.join(PM, "chinese-roberta-wwm-ext-large")}),
    ("2-get-hubert-wav32k.py", {"cnhubert_base_dir": os.path.join(PM, "chinese-hubert-base")}),
    (
        "3-get-semantic.py",
        {
            "pretrained_s2G": os.path.join(PM, "s2G488k.pth"),  # gin_channels=512, 与 s2.json 匹配
            "s2config_path": os.path.join(V4, "GPT_SoVITS", "configs", "s2.json"),
        },
    ),
]

ONLY = sys.argv[1] if len(sys.argv) > 1 else None
if ONLY:
    steps = [s for s in steps if ONLY in s[0]]
    print("[only]", [s[0] for s in steps])

for script, extra in steps:
    env = dict(base)
    env.update(extra)
    print("/n" + "=" * 55)
    print(">>> RUN", script)
    print("=" * 55, flush=True)
    r = subprocess.run([PY, os.path.join(PD, script)], cwd=V4, env=env)
    if r.returncode != 0:
        print("!!! FAILED:", script, "rc=", r.returncode)
        sys.exit(1)
    if script == "1-get-text.py":
        src, dst = os.path.join(OPT, "2-name2text-0.txt"), os.path.join(OPT, "2-name2text.txt")
    elif script == "3-get-semantic.py":
        src, dst = os.path.join(OPT, "6-name2semantic-0.tsv"), os.path.join(OPT, "6-name2semantic.tsv")
    else:
        continue
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("[merge]", os.path.basename(src), "->", os.path.basename(dst))

print("/n===== V4 预处理产物 =====")
for n in ("2-name2text.txt", "6-name2semantic.tsv"):
    p = os.path.join(OPT, n)
    print("  %-22s %s" % (n, sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else "MISSING"))
for n in ("4-cnhubert", "5-wav32k"):
    p = os.path.join(OPT, n)
    print("  %-22s %s" % (n, len(os.listdir(p)) if os.path.isdir(p) else "MISSING"))
