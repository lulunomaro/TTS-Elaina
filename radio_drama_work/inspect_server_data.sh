#!/bin/bash
set -u
for f in /root/autodl-tmp/ireina_v2/data/*.list; do
  printf '%s ' "$f"
  wc -l < "$f"
done
find /root/autodl-tmp/exp -maxdepth 3 -type f /( -name '2-name2text.txt' -o -name '6-name2semantic.tsv' /) | sort | tail -20
tail -20 /root/autodl-tmp/train_ireina_s2_balanced48_v3.log 2>/dev/null || true
