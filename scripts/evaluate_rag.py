"""Evaluate labeled RAG retrieval rows with hit-rate and MRR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no retrieval rows found in {path}")
    return rows


def score_retrieval(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")

    details = []
    hit_at_k = 0
    hit_at_1 = 0
    reciprocal_rank_sum = 0.0
    for row in rows:
        relevant = {str(value) for value in row.get("relevant_doc_ids", [])}
        retrieved = row.get("retrieved") or []
        retrieved_ids = [str(item.get("doc_id") or "") for item in retrieved]
        first_rank = next(
            (index for index, doc_id in enumerate(retrieved_ids, start=1) if doc_id in relevant),
            None,
        )
        hit = first_rank is not None and first_rank <= k
        if hit:
            hit_at_k += 1
            reciprocal_rank_sum += 1.0 / first_rank
        if first_rank == 1:
            hit_at_1 += 1
        details.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "relevant_doc_ids": sorted(relevant),
                "retrieved_doc_ids": retrieved_ids,
                "first_relevant_rank": first_rank,
                "hit_at_k": hit,
            }
        )

    total = len(rows)
    return {
        "task": "rag_retrieval",
        "total": total,
        "k": k,
        "hit_at_1": round(hit_at_1 / total, 4) if total else None,
        "hit_rate_at_k": round(hit_at_k / total, 4) if total else None,
        "mrr_at_k": round(reciprocal_rank_sum / total, 4) if total else None,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    report = score_retrieval(rows, args.k)
    report["retrieval_file"] = str(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
