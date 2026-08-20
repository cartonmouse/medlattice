"""Convert normalized CMB records into chat-style SFT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.prompting import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/cmb-exam-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cmb-exam-v1"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def to_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("reference_answer")
    choices = row.get("choices")
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"{row.get('id', '<unknown>')} has no reference answer")
    if not isinstance(choices, dict) or answer not in choices:
        raise ValueError(f"{row.get('id', '<unknown>')} has an invalid reference answer")

    return {
        "id": row["id"],
        "messages": build_messages(row, assistant_content=answer),
        "reference_answer": answer,
        "task_type": row.get("task_type", "multiple_choice"),
        "source": row.get("source"),
        "source_split": row.get("source_split"),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "messages",
        "assistant_target": "reference_answer",
        "explanations_used": False,
        "splits": {},
    }

    for split in ("train", "val"):
        source_path = args.input_dir / f"{split}.jsonl"
        rows = [to_sft_record(row) for row in load_jsonl(source_path)]
        output_path = args.output_dir / f"sft_{split}.jsonl"
        written = write_jsonl(output_path, rows)
        report["splits"][split] = {
            "input": str(source_path),
            "output": str(output_path),
            "written": written,
            "sha256": sha256_file(output_path),
        }

    metadata_path = args.output_dir / "sft_metadata.json"
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
