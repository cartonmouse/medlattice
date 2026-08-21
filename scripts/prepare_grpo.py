"""Prepare local CMB train-only records for a verifiable GRPO experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.grpo_data import build_grpo_record, split_grpo_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/cmb-exam-v1/train.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/cmb-grpo-v1")
    )
    parser.add_argument(
        "--final-val", type=Path, default=Path("data/processed/cmb-exam-v1/val.jsonl")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--max-input-samples", type=int)
    return parser.parse_args()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


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


def ids_from_jsonl(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in load_jsonl(path)}


def main() -> None:
    args = parse_args()
    if args.eval_size <= 0:
        raise ValueError("eval-size must be positive")
    raw_rows = load_jsonl(args.input, args.max_input_samples)
    if any(str(row.get("source_split", "")) != "train" for row in raw_rows):
        raise ValueError("GRPO preparation only accepts rows from source_split=train")

    records = [build_grpo_record(row) for row in raw_rows]
    train_records, eval_records = split_grpo_records(records, args.eval_size, args.seed)

    output_dir = args.output_dir
    train_path = output_dir / "grpo_train.jsonl"
    eval_path = output_dir / "grpo_eval.jsonl"
    train_count = write_jsonl(train_path, train_records)
    eval_count = write_jsonl(eval_path, eval_records)

    grpo_ids = {record["id"] for record in records}
    final_val_ids = ids_from_jsonl(args.final_val)
    metadata = {
        "format": "conversational_grpo_exact_match",
        "reward_functions": ["choice_exact_match_reward", "choice_format_reward"],
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "input_rows": len(raw_rows),
        "grpo_train_rows": train_count,
        "grpo_eval_rows": eval_count,
        "seed": args.seed,
        "eval_size": args.eval_size,
        "source_split_required": "train",
        "final_val_file": str(args.final_val),
        "final_val_rows": len(final_val_ids),
        "final_val_overlap_count": len(grpo_ids & final_val_ids),
        "final_cmb_val_is_not_used_for_grpo": True,
        "raw_data_committed_to_github": False,
        "outputs": {"train": str(train_path), "eval": str(eval_path)},
        "sha256": {"train": sha256_file(train_path), "eval": sha256_file(eval_path)},
    }
    metadata_path = output_dir / "grpo_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
