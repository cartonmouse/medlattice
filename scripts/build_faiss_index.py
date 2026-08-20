"""Build an optional FAISS HNSW index from the JSON dense index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.faiss_store import FaissVectorStore
from qwen_medical_qa.rag_embedding import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("outputs/rag-embedding-v1/index.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/faiss-v1/index.faiss"))
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=64)
    args = parser.parse_args()

    retriever = DenseRetriever.load(args.index)
    started = time.perf_counter()
    store = FaissVectorStore.build(
        path=args.output,
        chunks=retriever.chunks,
        embeddings=retriever.embeddings,
        model_name=retriever.model_name,
        query_instruction=retriever.query_instruction,
        max_length=retriever.max_length,
        pooling=retriever.pooling,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    print(
        json.dumps(
            {
                "backend": "faiss-hnsw",
                "index_type": store.index_type,
                "chunks": store.count,
                "dimension": store.dimension,
                "ef_search": args.ef_search,
                "build_seconds": round(time.perf_counter() - started, 4),
                "output": str(args.output),
                "metadata": f"{args.output}.meta.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
