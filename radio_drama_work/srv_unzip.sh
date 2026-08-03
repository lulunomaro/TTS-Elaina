#!/bin/bash
set -e
cd /root/autodl-tmp
rm -rf dataset_v1
which unzip >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq unzip)
unzip -q dataset_v1.zip -d /root/autodl-tmp/
echo "=== 解压结果 ==="
ls -d /root/autodl-tmp/dataset_v1
echo "wavs: $(ls /root/autodl-tmp/dataset_v1/wavs | wc -l) files"
echo "list: $(wc -l < /root/autodl-tmp/dataset_v1/dataset.list) lines"
du -sh /root/autodl-tmp/dataset_v1
echo "=== 首行 ==="
head -1 /root/autodl-tmp/dataset_v1/dataset.list
echo "=== 路径有效性抽查 ==="
head -3 /root/autodl-tmp/dataset_v1/dataset.list | cut -d'|' -f1 | while read f; do
  [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
done
echo "=== 采样率抽查 ==="
for f in $(ls /root/autodl-tmp/dataset_v1/wavs/*.wav | head -2); do
  ffprobe -v error -show_entries stream=sample_rate,channels,duration -of csv=p=0 "$f"
done
df -h /root/autodl-tmp | tail -1
