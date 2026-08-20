"""Retrieve and optionally BM25-rerank candidates from a FAISS HNSW index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.faiss_store import FaissVectorStore
from qwen_medical_qa.neural_reranker import TransformerCrossEncoderReranker
from qwen_medical_qa.rag import build_rag_prompt
from qwen_medical_qa.rag_embedding import TransformerTextEncoder
from qwen_medical_qa.reranker import BM25Reranker, describe_reranker


def read_queries(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no queries found in {path}")
    for row in rows:
        row["id"] = str(row.get("id") or "").strip()
        row["question"] = str(row.get("question") or "").strip()
        if not row["id"] or not row["question"]:
            raise ValueError("each query needs id and question")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("outputs/faiss-v1/index.faiss"))
    parser.add_argument("--input", type=Path, default=Path("data/rag_demo/retrieval_benchmark.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/faiss-v1/retrieval.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--reranker", choices=("none", "bm25", "neural"), default="bm25")
    parser.add_argument(
        "--reranker-model",
        default=TransformerCrossEncoderReranker.DEFAULT_MODEL_NAME,
    )
    parser.add_argument("--reranker-device", default="auto")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.candidate_k < args.top_k:
        raise ValueError("candidate-k must be greater than or equal to top-k")

    store = FaissVectorStore(args.index)
    rows = read_queries(args.input)
    encoder = TransformerTextEncoder(
        model_name=store.model_name,
        device=args.device,
        max_length=store.max_length,
        batch_size=args.batch_size,
        query_instruction=store.query_instruction,
        local_files_only=args.local_files_only,
    )
    embeddings = encoder.encode([row["question"] for row in rows], is_query=True)
    del encoder
    if args.reranker == "neural":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.reranker == "bm25":
        reranker = BM25Reranker()
    elif args.reranker == "neural":
        reranker = TransformerCrossEncoderReranker(
            model_name=args.reranker_model,
            device=args.reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            local_files_only=args.local_files_only,
        )
    else:
        reranker = None
    output_rows = []
    for row, embedding in zip(rows, embeddings):
        candidates = store.search(
            embedding,
            top_k=args.candidate_k,
            min_score=args.min_score,
        )
        reranker_latency_ms = None
        if reranker is None:
            results = candidates[: args.top_k]
        else:
            started = time.perf_counter()
            results = reranker.rerank(row["question"], candidates, top_k=args.top_k)
            reranker_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "relevant_doc_ids": row.get("relevant_doc_ids", []),
                "expected_answer": row.get("expected_answer"),
                "expected_abstain": row.get("expected_abstain"),
                "retriever": "faiss-hnsw",
                "reranker": args.reranker,
                **describe_reranker(reranker),
                "reranker_latency_ms": reranker_latency_ms,
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
                "backend": "faiss-hnsw",
                "reranker": args.reranker,
                **describe_reranker(reranker),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
