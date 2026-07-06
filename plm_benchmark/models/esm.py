"""GPU selection helper shared by the training pipeline."""

from __future__ import annotations

import os


def set_gpu(gpu_index: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index - 1)
