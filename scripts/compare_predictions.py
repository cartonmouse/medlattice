"""Compare two prediction JSONL files and summarize answer drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.prediction_compare import compare_prediction_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "left_file": str(args.left),
        "right_file": str(args.right),
        **compare_prediction_rows(read_jsonl(args.left), read_jsonl(args.right)),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
