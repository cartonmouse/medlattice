"""Evaluate generated RAG answers for retrieval, answer text, and citations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag_qa import score_rag_answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = score_rag_answers(rows, k=args.k)
    report["answer_file"] = str(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
