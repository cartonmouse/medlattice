"""Summarize latency and generation metrics from baseline JSONL output."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile without interpolating tiny runs."""
    ordered = sorted(values)
    rank = max(1, int((len(ordered) * fraction) + 0.999999))
    return ordered[min(rank, len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No rows found")

    latencies = [float(row["latency_ms"]) for row in rows]
    speeds = [float(row["tokens_per_second"]) for row in rows if row.get("tokens_per_second")]
    summary = {
        "samples": len(rows),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95_nearest_rank": round(percentile(latencies, 0.95), 2),
        },
        "tokens_per_second": {
            "mean": round(statistics.mean(speeds), 2) if speeds else None,
            "median": round(statistics.median(speeds), 2) if speeds else None,
        },
        "peak_cuda_allocated_mb": max(
            (row["peak_cuda_allocated_mb"] for row in rows if row.get("peak_cuda_allocated_mb") is not None),
            default=None,
        ),
        "peak_cuda_reserved_mb": max(
            (row["peak_cuda_reserved_mb"] for row in rows if row.get("peak_cuda_reserved_mb") is not None),
            default=None,
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

