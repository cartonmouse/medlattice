"""Retrieve and optionally BM25-rerank candidates from a SQLite vector store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import build_rag_prompt
from qwen_medical_qa.rag_embedding import TransformerTextEncoder
from qwen_medical_qa.reranker import BM25Reranker
from qwen_medical_qa.vector_store import SqliteVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("outputs/vector-store-v0/index.sqlite"))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rag_demo/retrieval_benchmark.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/vector-store-v0/retrieval.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--reranker", choices=("none", "bm25"), default="bm25")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not query_id or not question:
            raise ValueError(f"query at line {line_number} needs id and question")
        row["id"] = query_id
        row["question"] = question
        rows.append(row)
    if not rows:
        raise ValueError(f"no queries found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.candidate_k < args.top_k:
        raise ValueError("candidate-k must be greater than or equal to top-k")
    store = SqliteVectorStore(args.store)
    rows = read_queries(args.input)
    encoder = TransformerTextEncoder(
        model_name=store.model_name,
        device=args.device,
        max_length=store.max_length,
        batch_size=args.batch_size,
        query_instruction=store.query_instruction,
        local_files_only=args.local_files_only,
    )
    reranker = BM25Reranker() if args.reranker == "bm25" else None
    embeddings = encoder.encode([row["question"] for row in rows], is_query=True)
    output_rows = []
    for row, embedding in zip(rows, embeddings):
        candidates = store.search(
            embedding,
            top_k=args.candidate_k,
            min_score=args.min_score,
        )
        if reranker is None:
            results = candidates[: args.top_k]
        else:
            results = reranker.rerank(row["question"], candidates, top_k=args.top_k)
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "relevant_doc_ids": row.get("relevant_doc_ids", []),
                "expected_answer": row.get("expected_answer"),
                "expected_abstain": row.get("expected_abstain"),
                "retriever": "sqlite-dense",
                "reranker": args.reranker,
                "model_name": store.model_name,
                "candidate_k": args.candidate_k,
                "retrieval_abstained": not bool(candidates),
                "retrieved": [result.to_dict() for result in results],
                "context": build_rag_prompt(row["question"], results),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "queries": len(output_rows),
                "backend": "sqlite-dense",
                "reranker": args.reranker,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
