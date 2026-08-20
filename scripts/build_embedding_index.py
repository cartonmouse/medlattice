"""Build a dense embedding index with a Chinese BGE model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import chunk_document, read_documents_jsonl
from qwen_medical_qa.rag_embedding import (
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERY_INSTRUCTION,
    DenseRetriever,
    TransformerTextEncoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rag_demo/knowledge_base.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rag-embedding-v1/index.json"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_documents_jsonl(args.input)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, args.chunk_size, args.chunk_overlap)
    ]
    encoder = TransformerTextEncoder(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        query_instruction=args.query_instruction,
        local_files_only=args.local_files_only,
    )
    embeddings = encoder.encode([chunk.text for chunk in chunks], is_query=False)
    retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
        model_name=args.model_name,
        query_instruction=args.query_instruction,
        max_length=args.max_length,
    )
    retriever.save(args.output)
    print(
        json.dumps(
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "model_name": args.model_name,
                "device": encoder.device,
                "dimension": retriever.dimension,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
