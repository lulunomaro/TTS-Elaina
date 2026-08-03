#!/bin/bash
echo "=== BEFORE ==="
df -h /root /root/autodl-tmp | grep -v ^Filesystem
echo
# 只清理明确的中间产物（保留 pkg，暂不动）
rm -rf /root/autodl-tmp/handoff_build
rm -f  /root/autodl-tmp/repo.tar.gz /root/autodl-tmp/scripts.tar /root/autodl-tmp/nltk_eng.zip
mkdir -p /root/autodl-tmp/exp /root/autodl-tmp/dataset_v1
echo "=== AFTER ==="
df -h /root /root/autodl-tmp | grep -v ^Filesystem
