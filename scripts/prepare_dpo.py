"""Build a local conversational preference dataset for DPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.dpo_data import (  # noqa: E402
    NEGATIVE_STRATEGY,
    PREFERENCE_SOURCE,
    build_preference_pairs,
    split_preference_pairs,
)


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


def main() -> None:
    args = parse_args()
    input_path = args.input_dir / "train.jsonl"
    rows = load_jsonl(input_path, args.max_input_samples)
    pairs = build_preference_pairs(rows, seed=args.seed)
    train_pairs, eval_pairs = split_preference_pairs(
        pairs,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    train_path = args.output_dir / "dpo_train.jsonl"
    eval_path = args.output_dir / "dpo_eval.jsonl"
    train_count = write_jsonl(train_path, train_pairs)
    eval_count = write_jsonl(eval_path, eval_pairs)
    metadata = {
        "format": "conversational_explicit_prompt_preference",
        "preference_source": PREFERENCE_SOURCE,
        "negative_strategy": NEGATIVE_STRATEGY,
        "seed": args.seed,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rows": len(rows),
        "dpo_train_rows": train_count,
        "dpo_eval_rows": eval_count,
        "final_cmb_val_is_not_used_for_dpo": True,
        "final_benchmark": str(args.input_dir / "val.jsonl"),
        "raw_data_committed_to_github": False,
        "outputs": {
            "train": str(train_path),
            "eval": str(eval_path),
        },
        "sha256": {
            "train": sha256_file(train_path),
            "eval": sha256_file(eval_path),
        },
    }
    metadata_path = args.output_dir / "dpo_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
