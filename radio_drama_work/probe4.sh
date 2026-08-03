#!/bin/bash
cd /root/GPT-SoVITS
PY=/root/miniconda3/envs/gptsovits/bin/python
echo "===== s2v2Pro.json 关键字段 ====="
$PY -c "import json;d=json.load(open('GPT_SoVITS/configs/s2v2Pro.json'));print(' train.batch_size',d['train'].get('batch_size'),'epochs',d['train'].get('epochs'));print(' train keys:',list(d['train'].keys()));print(' data keys:',list(d['data'].keys()));print(' gin_channels',d['model'].get('gin_channels'))"
echo
echo "===== s1longer-v2.yaml ====="
head -40 GPT_SoVITS/configs/s1longer-v2.yaml
echo
echo "===== webui.py: SoVITS 训练函数 open1Ba 片段 ====="
grep -n "def open1Ba" webui.py
sed -n "$(grep -n 'def open1Ba' webui.py | head -1 | cut -d: -f1),+45p" webui.py
