"""Persist a dense JSON index into a SQLite-backed vector store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag_embedding import DenseRetriever
from qwen_medical_qa.vector_store import SqliteVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("outputs/rag-embedding-v1/index.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vector-store-v0/index.sqlite"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = DenseRetriever.load(args.index)
    store = SqliteVectorStore.build(
        path=args.output,
        chunks=retriever.chunks,
        embeddings=retriever.embeddings,
        model_name=retriever.model_name,
        query_instruction=retriever.query_instruction,
        max_length=retriever.max_length,
        pooling=retriever.pooling,
    )
    print(
        json.dumps(
            {
                "backend": "sqlite-vector-store",
                "chunks": store.count,
                "dimension": store.dimension,
                "metric": "inner-product-on-l2-normalized-vectors",
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
