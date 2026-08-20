"""Print a small, machine-readable environment report for experiment logs."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from typing import Any


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def main() -> None:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "transformers": module_version("transformers"),
        "accelerate": module_version("accelerate"),
        "safetensors": module_version("safetensors"),
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu": [],
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["gpu"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_gib": round(props.total_memory / 2**30, 2),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
    except Exception as exc:  # Keep the report useful even before dependencies install.
        report["torch_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

