"""Prepare an auditable DPO v2 dataset from UltraMedical-Preference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.dpo_data import split_preference_pairs  # noqa: E402
from qwen_medical_qa.public_dpo_data import (  # noqa: E402
    PUBLIC_DATASET_ID,
    PUBLIC_DATASET_REVISION,
    build_public_preference_pairs,
    prompt_overlap,
    select_preference_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/public-dpo/ultramedical/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/public-dpo/ultramedical-v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--max-train-pairs", type=int, default=2000)
    parser.add_argument("--train-file", default="dev.json")
    parser.add_argument("--human-eval-file", default="test.json")
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON array")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path}[{index}] must be a JSON object")
        rows.append(row)
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


def label_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("label_type", "unknown")) for row in rows).items()))


def main() -> None:
    args = parse_args()
    train_source = args.input_dir / args.train_file
    human_source = args.input_dir / args.human_eval_file
    train_candidates = load_json_array(train_source)
    human_candidates = load_json_array(human_source)

    train_pairs, train_filter_stats = build_public_preference_pairs(
        train_candidates,
        source_file=args.train_file,
        source_split="dev",
        require_strict_score=True,
    )
    selected_pairs = select_preference_pairs(
        train_pairs,
        max_pairs=args.max_train_pairs,
        seed=args.seed,
    )
    if args.eval_size <= 0 or args.eval_size >= len(selected_pairs):
        raise ValueError(
            f"eval-size must be in [1, {len(selected_pairs) - 1}] for the selected train set"
        )
    dpo_train, dpo_eval = split_preference_pairs(
        selected_pairs,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    human_eval, human_filter_stats = build_public_preference_pairs(
        human_candidates,
        source_file=args.human_eval_file,
        source_split="test_human",
        require_strict_score=False,
        allowed_label_types={"human"},
    )
    overlap = prompt_overlap(selected_pairs, human_eval)
    if overlap:
        raise ValueError(
            f"train and human evaluation prompts overlap: {len(overlap)} normalized prompts"
        )

    output_dir = args.output_dir
    train_path = output_dir / "dpo_train.jsonl"
    eval_path = output_dir / "dpo_eval.jsonl"
    human_path = output_dir / "human_eval.jsonl"
    train_count = write_jsonl(train_path, dpo_train)
    eval_count = write_jsonl(eval_path, dpo_eval)
    human_count = write_jsonl(human_path, human_eval)
    metadata: dict[str, Any] = {
        "format": "conversational_explicit_prompt_preference",
        "dataset_id": PUBLIC_DATASET_ID,
        "dataset_revision": PUBLIC_DATASET_REVISION,
        "selection_policy": {
            "train_source": args.train_file,
            "train_requires_chosen_score_gt_rejected_score": True,
            "train_deduplicate_by_prompt_id": True,
            "train_max_pairs": args.max_train_pairs,
            "human_evaluation_source": args.human_eval_file,
            "human_evaluation_label_types": ["human"],
            "human_evaluation_is_not_used_for_training": True,
        },
        "seed": args.seed,
        "raw_data_committed_to_github": False,
        "raw_files": {
            "train_source": {
                "path": str(train_source),
                "sha256": sha256_file(train_source),
                "rows": len(train_candidates),
            },
            "human_source": {
                "path": str(human_source),
                "sha256": sha256_file(human_source),
                "rows": len(human_candidates),
            },
        },
        "filter_stats": {
            "train_candidates": train_filter_stats.to_dict(),
            "human_candidates": human_filter_stats.to_dict(),
        },
        "selected_train_candidates": len(selected_pairs),
        "label_counts": {
            "selected_train": label_counts(selected_pairs),
            "dpo_train": label_counts(dpo_train),
            "dpo_eval": label_counts(dpo_eval),
            "human_eval": label_counts(human_eval),
        },
        "prompt_overlap": {
            "selected_train_vs_human_eval": len(overlap),
        },
        "splits": {
            "dpo_train": train_count,
            "dpo_eval": eval_count,
            "human_eval": human_count,
        },
        "outputs": {
            "train": str(train_path),
            "eval": str(eval_path),
            "human_eval": str(human_path),
        },
    }
    metadata["output_sha256"] = {
        "train": sha256_file(train_path),
        "eval": sha256_file(eval_path),
        "human_eval": sha256_file(human_path),
    }
    metadata_path = output_dir / "dpo_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
