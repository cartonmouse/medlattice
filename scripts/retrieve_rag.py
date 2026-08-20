"""Retrieve top-k knowledge chunks and render citation-aware prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import TfidfRetriever, build_rag_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("outputs/rag-v0/index.json"))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rag_demo/retrieval_benchmark.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/rag-v0/retrieval.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
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
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    retriever = TfidfRetriever.load(args.index)
    rows = read_queries(args.input)
    output_rows = []
    for row in rows:
        results = retriever.search(row["question"], top_k=args.top_k)
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "relevant_doc_ids": row.get("relevant_doc_ids", []),
                "retrieved": [result.to_dict() for result in results],
                "context": build_rag_prompt(row["question"], results),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"queries": len(output_rows), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
