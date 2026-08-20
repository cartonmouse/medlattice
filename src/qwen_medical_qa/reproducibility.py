"""Small, explicit helpers for repeatable local model experiments."""

from __future__ import annotations

import os
import random
from typing import Any


def configure_reproducibility(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed common RNGs and optionally request deterministic torch algorithms.

    The environment variable is set before importing torch so CUDA libraries can
    observe it during initialization. ``warn_only=True`` keeps unsupported
    deterministic operators visible without making every local experiment fail.
    The returned manifest is JSON-serializable and can be stored with predictions.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    numpy_available = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_available = True
    except ImportError:
        pass

    torch_version = None
    cuda_available = False
    torch_deterministic_algorithms = None
    cudnn_deterministic = None
    cudnn_benchmark = None
    try:
        import torch

        torch_version = str(torch.__version__)
        torch.manual_seed(seed)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    except ImportError:
        pass

    return {
        "seed": seed,
        "deterministic_requested": bool(deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "numpy_available": numpy_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "torch_deterministic_algorithms": torch_deterministic_algorithms,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }
