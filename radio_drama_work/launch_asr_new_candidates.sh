#!/bin/bash
set -u
LOG=/root/autodl-tmp/asr_new_candidates.log
rm -f "$LOG"
nohup setsid bash /tmp/_wb_asr_new_candidates.sh >"$LOG" 2>&1 </dev/null &
disown || true
echo "started pid=$! log=$LOG"
