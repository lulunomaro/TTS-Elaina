# -*- coding: utf-8 -*-
"""V4 LoRA 训练到 e8 即停，随后产出 v2pro / V4 同句对比音频。

理由: V4 官方 webui 对 v3/v4 的 SoVITS LoRA 默认 epoch=2、上限 16，
源码注释「训太多=作死」。原编排器配了 40 轮(约5小时且易过拟合)，
故改为在 e8 权重落盘后立即终止训练进程，直接进入对比阶段。

流程:
  1. 轮询 SoVITS_weights_v4_merged/ 等待 *_e8_*.pth
  2. 文件大小稳定后，终止 run_v4_pipeline.py 与 s2_train_v3_lora.py 进程
  3. 用同一份合并 GPT (v2pro e40) 分别驱动 v2pro-e40 与 V4-e8 SoVITS 合成对比音频
"""
import io
import os
import sys
import time
import glob
import subprocess

import psutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

V2 = r"radio_drama_work/GPT-SoVITS-v2pro-20250604-nvidia50/GPT-SoVITS-v2pro-20250604-nvidia50"
V4 = r"radio_drama_work/GPT-SoVITS-v4-20250529-nvidia50/GPT-SoVITS-v4-20250529-nvidia50"
V2_PY = os.path.join(V2, "runtime", "python.exe")
V4_PY = os.path.join(V4, "runtime", "python.exe")
WORK = r"radio_drama_work/radio_drama_work"

V4_SAVE = os.path.join(V4, "SoVITS_weights_v4_merged")
OUT_DIR = os.path.join(WORK, "compare")
V2_GPT = os.path.join(V2, "GPT_weights_v2Pro_merged", "my_voice_merged-e40.ckpt")
LOG = os.path.join(WORK, "finish_e8.log")

TARGET_EPOCH = 8


def log(*a):
    msg = "[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "/n")


def run(cmd, cwd, label):
    log(">>> RUN", label)
    log("    ", " ".join(cmd))
    with open(LOG, "a", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        log("!!! FAILED", label, "rc=", p.returncode)
        return False
    log("<<< OK", label)
    return True


def find_e8():
    hits = glob.glob(os.path.join(V4_SAVE, "*_e%d_*.pth" % TARGET_EPOCH))
    return sorted(hits)[-1] if hits else None


def kill_training():
    """先杀编排器(父)，再杀 LoRA 训练进程(子)，避免父进程重启子进程。"""
    victims = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if "run_v4_pipeline.py" in cl:
            victims.append(("orchestrator", p))
        elif "s2_train_v3_lora.py" in cl:
            victims.append(("lora_train", p))
    victims.sort(key=lambda x: 0 if x[0] == "orchestrator" else 1)
    for tag, p in victims:
        try:
            log("终止进程", tag, "pid=", p.pid)
            p.terminate()
        except Exception as e:
            log("terminate 失败", p.pid, e)
    gone, alive = psutil.wait_procs([p for _, p in victims], timeout=25)
    for p in alive:
        try:
            log("强制 kill pid=", p.pid)
            p.kill()
        except Exception:
            pass
    log("已终止进程数:", len(victims))
    time.sleep(8)  # 等显存释放


# ---------- 1. 等待 e8 ----------
log("=== 等待 V4 LoRA e%d 权重落盘 ===" % TARGET_EPOCH)
log("监视目录:", V4_SAVE)
deadline = time.time() + 180 * 60
ck = None
while time.time() < deadline:
    ck = find_e8()
    if ck:
        s1 = os.path.getsize(ck)
        time.sleep(15)
        if os.path.getsize(ck) == s1 and s1 > 1024 * 1024:
            log("检测到 e%d 权重:" % TARGET_EPOCH, ck, "(%.1f MB)" % (s1 / 1024 / 1024))
            break
        ck = None
    time.sleep(30)

if not ck:
    log("!!! 等待 e%d 超时" % TARGET_EPOCH)
    sys.exit(1)

# ---------- 2. 停训 ----------
log("=== 停止 V4 训练(已达目标轮数，避免过拟合与无谓耗时) ===")
kill_training()

# ---------- 3. 定位 v2pro 权重 ----------
v2_list = sorted(glob.glob(os.path.join(V2, "SoVITS_weights_v2Pro_merged", "my_voice_merged_e40_*.pth")))
if not v2_list:
    log("!!! 未找到 v2pro e40 SoVITS 权重")
    sys.exit(1)
v2_sovits = v2_list[-1]
log("v2pro SoVITS:", v2_sovits)
log("V4    SoVITS:", ck)
log("共用 GPT    :", V2_GPT)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 4. 合成对比 ----------
ok1 = run([V2_PY, os.path.join(WORK, "infer_compare.py"), "v2Pro", V2_GPT, v2_sovits,
           os.path.join(OUT_DIR, "v2pro")], cwd=V2, label="infer_v2pro")
ok2 = run([V4_PY, os.path.join(WORK, "infer_compare.py"), "v4", V2_GPT, ck,
           os.path.join(OUT_DIR, "v4")], cwd=V4, label="infer_v4")

log("===== 完成: v2pro=%s  v4=%s =====" % (ok1, ok2))
log("对比音频目录:", OUT_DIR)
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.wav"))):
    log("  ", os.path.basename(f), "%.1f KB" % (os.path.getsize(f) / 1024))
