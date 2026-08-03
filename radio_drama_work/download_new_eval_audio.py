# -*- coding: utf-8 -*-
"""Download the latest paired-evaluation audio into the compact listen pack."""
import json
import os
import runpy

ROOT = r"radio_drama_work/compact_v1"
MANIFEST = r"radio_drama_work/paired_emo_eval.jsonl"
REMOTE_PY = r"radio_drama_work/remote.py"

remote = runpy.run_path(REMOTE_PY)
cli = remote["connect"]()
sftp = cli.open_sftp()
try:
    rows = [json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()]
    real_dir = os.path.join(ROOT, "dataset_v1", "wavs")
    os.makedirs(real_dir, exist_ok=True)
    for row in rows:
        rpath = row["real_path"]
        local = os.path.join(real_dir, os.path.basename(rpath))
        sftp.get(rpath, local)
        print("real", row["sample_id"])

    for rdir, local_dir in (
        ("/root/autodl-tmp/infer_paired_text_lr2_v1", "infer_paired_text_lr2_v1"),
        ("/root/autodl-tmp/infer_paired_text_lr2_e2_v1", "infer_paired_text_lr2_e2_v1"),
        ("/root/autodl-tmp/infer_paired_text_lr1e5_e2_v1", "infer_paired_text_lr1e5_e2_v1"),
    ):
        out_dir = os.path.join(ROOT, local_dir)
        os.makedirs(out_dir, exist_ok=True)
        for name in sftp.listdir(rdir):
            if name.endswith(".wav"):
                sftp.get(rdir + "/" + name, os.path.join(out_dir, name))
                print(local_dir, name)
finally:
    sftp.close()
    cli.close()
