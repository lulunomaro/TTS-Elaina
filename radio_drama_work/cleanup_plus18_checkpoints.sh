#!/bin/bash
set -euo pipefail
DIR=/root/autodl-tmp/exp/ireina_s2_balanced48_plus18_v1/logs_s2_v2Pro
find "$DIR" -maxdepth 1 -type f /( -name 'G_*.pth' -o -name 'D_*.pth' /) -print -delete
echo "remaining:"; find "$DIR" -maxdepth 1 -type f -printf '%f %s bytes/n' | sort
du -sh "$DIR"
