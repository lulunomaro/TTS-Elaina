# -*- coding: utf-8 -*-
"""对合并数据集跑 GPT-SoVITS 数据预处理四件套（单卡 i_part=0/all_parts=1）。

1-get-text.py          -> 2-name2text-0.txt   -> 合并为 2-name2text.txt
2-get-hubert-wav32k.py -> 4-cnhubert/ 5-wav32k/
2-get-sv.py            -> 7-sv_cn/
3-get-semantic.py      -> 6-name2semantic-0.tsv -> 合并为 6-name2semantic.tsv
"""
import io
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
PY = os.path.join(ROOT, "runtime", "python.exe")
PD = os.path.join(ROOT, "GPT_SoVITS", "prepare_datasets")
EXP = "my_voice_merged"
OPT = os.path.join(ROOT, "logs", EXP)
LIST = os.path.join(OPT, "merged.list")
PM = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models")

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
        "version": "v2Pro",
        "PYTHONIOENCODING": "utf-8",
    }
)

steps = [
    ("1-get-text.py", {"bert_pretrained_dir": os.path.join(PM, "chinese-roberta-wwm-ext-large")}),
    ("2-get-hubert-wav32k.py", {"cnhubert_base_dir": os.path.join(PM, "chinese-hubert-base")}),
    ("2-get-sv.py", {"sv_path": os.path.join(PM, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")}),
    (
        "3-get-semantic.py",
        {
            "pretrained_s2G": os.path.join(PM, "v2Pro", "s2Gv2Pro.pth"),
            "s2config_path": os.path.join(ROOT, "GPT_SoVITS", "configs", "s2v2Pro.json"),
        },
    ),
]

ONLY = sys.argv[1] if len(sys.argv) > 1 else None
if ONLY:
    steps = [s for s in steps if ONLY in s[0]]
    print("[only] 仅执行:", [s[0] for s in steps])

for script, extra in steps:
    env = dict(base)
    env.update(extra)
    print("/n" + "=" * 60)
    print(">>> RUN", script)
    print("=" * 60, flush=True)
    r = subprocess.run([PY, os.path.join(PD, script)], cwd=ROOT, env=env)
    if r.returncode != 0:
        print("!!! FAILED:", script, "rc=", r.returncode)
        sys.exit(1)

    # part 文件合并
    if script == "1-get-text.py":
        src, dst = os.path.join(OPT, "2-name2text-0.txt"), os.path.join(OPT, "2-name2text.txt")
    elif script == "3-get-semantic.py":
        src, dst = os.path.join(OPT, "6-name2semantic-0.tsv"), os.path.join(OPT, "6-name2semantic.tsv")
    else:
        continue
    if os.path.exists(src):
        data = open(src, encoding="utf-8").read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(data)
        print("[merge] %s -> %s (%d 行)" % (os.path.basename(src), os.path.basename(dst), data.count("/n")))

print("/n===== 预处理产物 =====")
for n in ("2-name2text.txt", "6-name2semantic.tsv"):
    p = os.path.join(OPT, n)
    print("  %-24s %s" % (n, sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else "MISSING"))
for n in ("4-cnhubert", "5-wav32k", "7-sv_cn"):
    p = os.path.join(OPT, n)
    print("  %-24s %s" % (n, len(os.listdir(p)) if os.path.isdir(p) else "MISSING"))
