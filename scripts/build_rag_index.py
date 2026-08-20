"""Build the dependency-free TF-IDF index used by the RAG v0 demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import TfidfRetriever, chunk_document, read_documents_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rag_demo/knowledge_base.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rag-v0/index.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_documents_jsonl(args.input)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, args.chunk_size, args.chunk_overlap)
    ]
    retriever = TfidfRetriever(chunks)
    retriever.save(args.output)
    summary = {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
