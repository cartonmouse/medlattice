"""Download and normalize the CMB-Exam dataset without committing raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "FreedomIntelligence/CMB"
DATASET_CONFIG = "CMB-Exam"
DECLARED_LICENSE = "Apache-2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cmb-exam-v1"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--val-limit", type=int, default=280)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        answer = "".join(str(item) for item in answer)
    return "".join(character for character in str(answer).strip().upper() if character.isalpha())


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError as exc:
            raise ValueError("option string is not valid JSON") from exc
    if not isinstance(options, dict):
        raise ValueError("option must be an object")
    normalized: dict[str, str] = {}
    for key, value in options.items():
        option_key = str(key).strip().upper()
        option_value = str(value).strip()
        if not option_key or not option_value:
            continue
        normalized[option_key] = option_value
    if len(normalized) < 2:
        raise ValueError("at least two non-empty options are required")
    return normalized


def normalize_record(raw: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError("question is empty")
    options = normalize_options(raw.get("option", raw.get("options")))
    answer = normalize_answer(raw.get("answer", ""))
    question_type = str(raw.get("question_type") or "")
    if len(answer) == 1:
        task_type = "multiple_choice"
    elif answer:
        task_type = "multiple_answer"
    elif "多项" in question_type:
        task_type = "multiple_answer"
    else:
        task_type = "multiple_choice"

    raw_id = str(raw.get("id", index)).strip()
    stable_id = f"cmb-exam-{split}-{raw_id}"
    return {
        "id": stable_id,
        "task_type": task_type,
        "question": question,
        "choices": options,
        "reference_answer": answer or None,
        "reference_explanation": raw.get("explanation"),
        "exam_type": raw.get("exam_type"),
        "exam_class": raw.get("exam_class"),
        "exam_subject": raw.get("exam_subject"),
        "question_type": raw.get("question_type"),
        "source": DATASET_ID,
        "license": DECLARED_LICENSE,
        "source_split": split,
    }


def is_single_choice(row: dict[str, Any]) -> bool:
    question_type = str(row.get("question_type") or "")
    return row["task_type"] == "multiple_choice" and "多项" not in question_type


def has_valid_reference_answer(row: dict[str, Any]) -> bool:
    answer = row.get("reference_answer")
    return answer is None or answer in row.get("choices", {})


def deterministic_limit(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in selected_indices]


def deterministic_reservoir(
    rows: Iterable[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Select a reproducible subset without materializing a large split."""
    if limit <= 0:
        return list(rows)

    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any]]] = []
    seen = 0
    for source_index, row in enumerate(rows):
        seen += 1
        if len(selected) < limit:
            selected.append((source_index, row))
            continue
        replacement = rng.randrange(seen)
        if replacement < limit:
            selected[replacement] = (source_index, row)

    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def load_cmb(cache_dir: Path | None) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    return load_dataset(DATASET_ID, DATASET_CONFIG, streaming=True, **kwargs)


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
    dataset = load_cmb(args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_report: dict[str, Any] = {}

    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    for split, limit in limits.items():
        if split not in dataset:
            raise KeyError(f"Expected split {split!r}; available: {list(dataset.keys())}")

        raw_rows = dataset[split]
        normalized: list[dict[str, Any]] = []
        skipped = Counter()
        raw_count = 0
        for index, raw in enumerate(raw_rows):
            raw_count = index + 1
            try:
                row = normalize_record(dict(raw), split, index)
            except (TypeError, ValueError, KeyError) as exc:
                skipped[type(exc).__name__] += 1
                continue
            if not row["reference_answer"] and split != "test":
                skipped["missing_answer"] += 1
                continue
            if not is_single_choice(row):
                skipped["multiple_answer"] += 1
                continue
            if not has_valid_reference_answer(row):
                skipped["answer_not_in_choices"] += 1
                continue
            normalized.append(row)

        split_seed = args.seed + sum(ord(character) for character in split)
        selected = deterministic_reservoir(normalized, limit, split_seed)
        output_path = args.output_dir / f"{split}.jsonl"
        written = write_jsonl(output_path, selected)
        split_report[split] = {
            "raw": raw_count if normalized or skipped else 0,
            "single_choice_valid": len(normalized),
            "written": written,
            "limit": limit,
            "labeled": sum(1 for row in selected if row["reference_answer"]),
            "unlabeled": sum(1 for row in selected if not row["reference_answer"]),
            "skipped": dict(skipped),
            "file": str(output_path),
            "sha256": sha256_file(output_path),
        }

    metadata = {
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "declared_license": DECLARED_LICENSE,
        "seed": args.seed,
        "single_choice_only": True,
        "splits": split_report,
        "raw_data_committed_to_github": False,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
