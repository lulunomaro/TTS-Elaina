# -*- coding: utf-8 -*-
"""V4 训练 + v2pro/V4 同句对比编排器。

流程:
  1. 轮询等待 v2pro 合并版 S2 训练完成 (s2_merged.log 出现 "training done")
  2. 在 V4 包内跑合并数据预处理 (prepare_v4.py)
  3. 生成 V4 s2 LoRA 训练配置并启动训练 (s2_train_v3_lora.py)
  4. 训练完成后，用 infer_compare.py 分别合成 v2pro 与 V4 的对比音频
     (复用同一份合并 GPT，仅比 SoVITS 差异，公平对比)

全程日志写入 radio_drama_work/v4_pipeline.log
"""
import io
import json
import os
import sys
import time
import glob
import subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

V2 = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
V4 = r"radio_drama_work/GPT-SoVITS-v4-20250529-nvidia50/GPT-SoVITS-v4-20250529-nvidia50"
V2_PY = os.path.join(V2, "runtime", "python.exe")
V4_PY = os.path.join(V4, "runtime", "python.exe")
WORK = r"radio_drama_work/radio_drama_work"

S2_LOG = os.path.join(V2, "logs", "my_voice_merged", "tmp", "s2_merged.log")
V4_CFG = os.path.join(V4, "logs", "my_voice_merged", "tmp", "s2_merged_v4.json")
V4_SAVE = os.path.join(V4, "SoVITS_weights_v4_merged")
OUT_DIR = os.path.join(WORK, "compare")
V2_GPT = os.path.join(V2, "GPT_weights_v2Pro_merged", "my_voice_merged-e40.ckpt")

LOG = os.path.join(WORK, "v4_pipeline.log")


def log(*a):
    msg = "[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "/n")


def run(cmd, cwd, label):
    log(">>> RUN", label, "| cwd=", cwd)
    log("    ", " ".join(cmd))
    with open(LOG, "a", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        log("!!! FAILED", label, "rc=", p.returncode)
        sys.exit(1)
    log("<<< OK", label)


# ---------- 1. 等待 S2 v2pro 合并版训练完成 ----------
log("=== STAGE 1: 等待 v2pro 合并版 S2 训练完成 ===")
deadline = time.time() + 150 * 60  # 最多等 150 分钟
done = False
while time.time() < deadline:
    if os.path.exists(S2_LOG) and "training done" in open(S2_LOG, encoding="utf-8", errors="ignore").read():
        done = True
        break
    # 若 e40 SoVITS 权重已落盘且进程结束，也视为完成
    e40 = glob.glob(os.path.join(V2, "SoVITS_weights_v2Pro_merged", "my_voice_merged_e40_*.pth"))
    if e40 and "Epoch: 40" in open(S2_LOG, encoding="utf-8", errors="ignore").read():
        done = True
        break
    time.sleep(60)
if not done:
    log("!!! 等待 S2 超时，退出")
    sys.exit(1)
log("=== S2 v2pro 合并版训练已完成 ===")

# ---------- 2. V4 预处理 ----------
log("=== STAGE 2: V4 合并数据预处理 ===")
run([V4_PY, os.path.join(WORK, "prepare_v4.py")], cwd=V4, label="prepare_v4")

# ---------- 3. 生成 V4 配置并训练 ----------
log("=== STAGE 3: 生成 V4 s2 LoRA 配置 ===")
with open(os.path.join(V4, "GPT_SoVITS", "configs", "s2.json"), encoding="utf-8") as f:
    cfg = json.load(f)
s2_dir = "logs/my_voice_merged"
cfg["train"]["batch_size"] = 3
cfg["train"]["epochs"] = 40
cfg["train"]["text_low_lr_rate"] = 0.4
cfg["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
cfg["train"]["pretrained_s2D"] = ""
cfg["train"]["if_save_latest"] = 1
cfg["train"]["if_save_every_weights"] = 1
cfg["train"]["save_every_epoch"] = 4
cfg["train"]["gpu_numbers"] = "0"
cfg["train"]["grad_ckpt"] = False
cfg["train"]["lora_rank"] = 32
cfg["model"]["version"] = "v4"
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = s2_dir
cfg["save_weight_dir"] = V4_SAVE
cfg["name"] = "my_voice_merged"
cfg["version"] = "v4"
os.makedirs(V4_SAVE, exist_ok=True)
os.makedirs(os.path.dirname(V4_CFG), exist_ok=True)
with open(V4_CFG, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
log("V4 配置已写入", V4_CFG)

log("=== STAGE 4: 启动 V4 LoRA 训练 (epochs=40, lora_rank=32) ===")
run([V4_PY, "-s", "GPT_SoVITS/s2_train_v3_lora.py", "--config", V4_CFG], cwd=V4, label="s2_train_v3_lora")

# ---------- 5. 找出最终权重并对比 ----------
log("=== STAGE 5: 定位最终权重 ===")
v2_sovits_list = sorted(glob.glob(os.path.join(V2, "SoVITS_weights_v2Pro_merged", "my_voice_merged_e40_*.pth")))
v4_sovits_list = sorted(glob.glob(os.path.join(V4_SAVE, "my_voice_merged_e40_*_l32.pth")))
if not v2_sovits_list:
    log("!!! 未找到 v2pro e40 SoVITS 权重")
    sys.exit(1)
if not v4_sovits_list:
    log("!!! 未找到 V4 e40 LoRA 权重")
    sys.exit(1)
v2_sovits = v2_sovits_list[-1]
v4_sovits = v4_sovits_list[-1]
log("v2pro SoVITS:", v2_sovits)
log("V4    SoVITS:", v4_sovits)
log("共用 GPT:", V2_GPT)

os.makedirs(OUT_DIR, exist_ok=True)
log("=== STAGE 6: 合成 v2pro 对比音频 ===")
run([V2_PY, os.path.join(WORK, "infer_compare.py"), "v2Pro", V2_GPT, v2_sovits,
     os.path.join(OUT_DIR, "v2pro")], cwd=V2, label="infer_v2pro")
log("=== STAGE 7: 合成 V4 对比音频 ===")
run([V4_PY, os.path.join(WORK, "infer_compare.py"), "v4", V2_GPT, v4_sovits,
     os.path.join(OUT_DIR, "v4")], cwd=V4, label="infer_v4")

log("===== 全部完成 =====")
log("对比音频目录:", OUT_DIR)
