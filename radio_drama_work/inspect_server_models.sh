#!/bin/bash
set -u
find /root/.cache /root/autodl-tmp -maxdepth 6 -type f /( -name 'pytorch_model.bin' -o -name 'model.safetensors' -o -name '*.ckpt' /) 2>/dev/null | grep -Ei 'wav2vec|wavlm|mos|unispeech|hubert|speech' | head -60
