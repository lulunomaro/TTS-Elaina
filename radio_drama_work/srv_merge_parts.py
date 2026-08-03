# -*- coding: utf-8 -*-
"""合并预处理分片产物（官方 webui 在多进程结束后做的这一步，脚本方式必须手动补）。
  2-name2text-{i}.txt  -> 2-name2text.txt        (无表头)
  6-name2semantic-{i}.tsv -> 6-name2semantic.tsv (需加表头 item_name/tsemantic_audio)
"""
import os
import sys
import glob

EXP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/exp/ireina_v1"

# ---- 2-name2text ----
parts = sorted(glob.glob(os.path.join(EXP_DIR, "2-name2text-*.txt")))
if parts:
    opt = []
    for p in parts:
        with open(p, "r", encoding="utf8") as f:
            opt += f.read().strip("/n").split("/n")
    opt = [x for x in opt if x.strip()]
    dst = os.path.join(EXP_DIR, "2-name2text.txt")
    with open(dst, "w", encoding="utf8") as f:
        f.write("/n".join(opt) + "/n")
    for p in parts:
        os.remove(p)
    print("2-name2text.txt      : %d lines (merged from %d part)" % (len(opt), len(parts)))
else:
    dst = os.path.join(EXP_DIR, "2-name2text.txt")
    print("2-name2text.txt      : already merged (%d lines)" %
          (len(open(dst, encoding="utf8").read().strip("/n").split("/n")) if os.path.exists(dst) else 0))

# ---- 6-name2semantic ----
parts = sorted(glob.glob(os.path.join(EXP_DIR, "6-name2semantic-*.tsv")))
if parts:
    opt = ["item_name/tsemantic_audio"]
    for p in parts:
        with open(p, "r", encoding="utf8") as f:
            opt += f.read().strip("/n").split("/n")
    opt = [x for x in opt if x.strip()]
    dst = os.path.join(EXP_DIR, "6-name2semantic.tsv")
    with open(dst, "w", encoding="utf8") as f:
        f.write("/n".join(opt) + "/n")
    for p in parts:
        os.remove(p)
    print("6-name2semantic.tsv  : %d lines incl. header (merged from %d part)" % (len(opt), len(parts)))
else:
    dst = os.path.join(EXP_DIR, "6-name2semantic.tsv")
    print("6-name2semantic.tsv  : already merged (%d lines)" %
          (len(open(dst, encoding="utf8").read().strip("/n").split("/n")) if os.path.exists(dst) else 0))

print("/nfiles in exp dir:")
for fn in sorted(os.listdir(EXP_DIR)):
    p = os.path.join(EXP_DIR, fn)
    if os.path.isdir(p):
        print("  d %-24s %d files" % (fn, len(os.listdir(p))))
    else:
        print("  - %-24s %d bytes" % (fn, os.path.getsize(p)))
