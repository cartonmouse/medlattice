"""Compare retrieval result rankings without exposing retrieved text."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{label} contains a row without id")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate id: {row_id}")
        indexed[row_id] = row
    if not indexed:
        raise ValueError(f"{label} contains no rows")
    return indexed


def compare_retrieval_rows(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    """Compare ranking overlap for two retrieval JSONL collections."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    left = _index(left_rows, "left")
    right = _index(right_rows, "right")
    if set(left) != set(right):
        raise ValueError("retrieval query IDs differ")

    exact_ranking = 0
    exact_set = 0
    overlap_sum = 0.0
    for row_id in sorted(left):
        left_ids = [
            str(item.get("chunk_id") or "")
            for item in (left[row_id].get("retrieved") or [])[:top_k]
        ]
        right_ids = [
            str(item.get("chunk_id") or "")
            for item in (right[row_id].get("retrieved") or [])[:top_k]
        ]
        left_set = set(left_ids)
        right_set = set(right_ids)
        exact_ranking += int(left_ids == right_ids)
        exact_set += int(left_set == right_set)
        overlap_sum += len(left_set & right_set) / max(1, top_k)

    total = len(left)
    return {
        "total": total,
        "top_k": top_k,
        "exact_ranking_match": exact_ranking,
        "exact_set_match": exact_set,
        "mean_set_overlap": round(overlap_sum / total, 6),
        "left_backend": left[next(iter(left))].get("retriever"),
        "right_backend": right[next(iter(right))].get("retriever"),
    }
