#!/usr/bin/env python3
import torch

PATH = "/root/autodl-tmp/weights/SoVITS_ireina_s2_balanced48_emo20_dlr2e5_b16_v1/ireina_s2_balanced48_emo20_dlr2e5_b16_v1_e2_s40.pth"
for weights_only in (True, False):
    try:
        obj = torch.load(PATH, map_location="cpu", weights_only=weights_only)
        print(f"weights_only={weights_only} type={type(obj).__name__} keys={list(obj.keys())[:8] if hasattr(obj, 'keys') else 'none'}")
    except Exception as exc:
        print(f"weights_only={weights_only} FAIL {type(exc).__name__}: {exc}")
