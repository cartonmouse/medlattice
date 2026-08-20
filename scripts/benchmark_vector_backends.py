"""Benchmark SQLite scan and FAISS HNSW search on the same query embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.faiss_store import FaissVectorStore
from qwen_medical_qa.rag_embedding import TransformerTextEncoder
from qwen_medical_qa.vector_store import SqliteVectorStore


def read_queries(path: Path) -> list[dict[str, str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no queries found in {path}")
    result = []
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not row_id or not question:
            raise ValueError("each query needs id and question")
        result.append({"id": row_id, "question": question})
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return ordered[index]


def benchmark_store(store: Any, embeddings: list[list[float]], top_k: int) -> dict[str, Any]:
    timings = []
    for embedding in embeddings:
        started = time.perf_counter()
        store.search(embedding, top_k=top_k)
        timings.append((time.perf_counter() - started) * 1000)
    path = Path(store.path)
    size_bytes = path.stat().st_size
    metadata_path = getattr(store, "metadata_path", None)
    if metadata_path is not None and Path(metadata_path).exists():
        size_bytes += Path(metadata_path).stat().st_size
    return {
        "count": store.count,
        "dimension": store.dimension,
        "index_bytes_including_metadata": size_bytes,
        "mean_search_ms": round(statistics.mean(timings), 4),
        "p50_search_ms": round(percentile(timings, 0.50), 4),
        "p95_search_ms": round(percentile(timings, 0.95), 4),
        "max_search_ms": round(max(timings), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--faiss", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    sqlite_store = SqliteVectorStore(args.sqlite)
    faiss_store = FaissVectorStore(args.faiss)
    if sqlite_store.model_name != faiss_store.model_name:
        raise ValueError("SQLite and FAISS stores use different embedding models")
    queries = read_queries(args.input)
    encoder = TransformerTextEncoder(
        model_name=sqlite_store.model_name,
        device=args.device,
        max_length=sqlite_store.max_length,
        batch_size=args.batch_size,
        query_instruction=sqlite_store.query_instruction,
        local_files_only=args.local_files_only,
    )
    started = time.perf_counter()
    embeddings = encoder.encode([row["question"] for row in queries], is_query=True)
    embedding_seconds = time.perf_counter() - started
    result = {
        "queries": len(queries),
        "top_k": args.top_k,
        "embedding_model": sqlite_store.model_name,
        "embedding_seconds": round(embedding_seconds, 4),
        "search_timing_excludes_embedding": True,
        "backends": {
            "sqlite_scan": benchmark_store(sqlite_store, embeddings, args.top_k),
            "faiss_hnsw": benchmark_store(faiss_store, embeddings, args.top_k),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
