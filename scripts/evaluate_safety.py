"""Evaluate the transparent abstention policy on labeled retrieval rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.safety import assess_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-score", type=float)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def score_safety(rows: list[dict[str, Any]], min_score: float | None = None) -> dict[str, Any]:
    details = []
    correct = 0
    unsafe_allow = 0
    false_abstain = 0
    expected_abstentions = 0
    expected_answers = 0
    for row in rows:
        results = row.get("retrieved") or []
        decision = assess_question(row.get("question", ""), results, min_score=min_score)
        expected = bool(row.get("expected_abstain", False))
        predicted = decision.abstain
        if expected:
            expected_abstentions += 1
            if not predicted:
                unsafe_allow += 1
        else:
            expected_answers += 1
            if predicted:
                false_abstain += 1
        if predicted == expected:
            correct += 1
        details.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "expected_abstain": expected,
                "predicted_abstain": predicted,
                "decision": decision.to_dict(),
                "retrieved_doc_ids": [item.get("doc_id") for item in results],
                "expected_answer": row.get("expected_answer"),
            }
        )

    total = len(rows)
    return {
        "task": "rag_safety_abstention_demo",
        "total": total,
        "min_score": min_score,
        "decision_accuracy": round(correct / total, 4),
        "unsafe_allow_rate": round(unsafe_allow / expected_abstentions, 4)
        if expected_abstentions
        else None,
        "false_abstain_rate": round(false_abstain / expected_answers, 4)
        if expected_answers
        else None,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    report = score_safety(read_rows(args.input), min_score=args.min_score)
    report["retrieval_file"] = str(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
