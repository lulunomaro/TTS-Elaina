# -*- coding: utf-8 -*-
"""Keep local mining GPU-first and friendly to interactive CAD workloads."""
from __future__ import annotations

import os


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_thread_environment(max_threads: int = 2) -> None:
    value = str(max(1, int(max_threads)))
    for name in THREAD_ENV_VARS:
        os.environ[name] = value
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def apply_cpu_friendly_policy(max_threads: int = 2) -> dict:
    """Limit CPU worker pools and lower this process below normal priority."""
    set_thread_environment(max_threads)
    priority = "unchanged"
    if os.name == "nt":
        try:
            import ctypes

            below_normal_priority_class = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.kernel32.SetPriorityClass(handle, below_normal_priority_class):
                priority = "below_normal"
        except Exception:
            priority = "unchanged"
    else:
        try:
            os.nice(10)
            priority = "nice+10"
        except OSError:
            priority = "unchanged"

    try:
        import torch

        torch.set_num_threads(max(1, int(max_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(max(1, int(max_threads)))
    except Exception:
        pass
    return {
        "cpu_threads": max(1, int(max_threads)),
        "process_priority": priority,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }


# These variables must be set before NumPy/MKL/PyTorch are imported by callers.
set_thread_environment(2)
