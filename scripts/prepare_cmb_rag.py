"""Prepare a leakage-aware closed-domain CMB-Exam retrieval corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/cmb-exam-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cmb-rag-v1"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def normalize_for_dedup(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def render_question(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    choices = row.get("choices") or {}
    if not question or not isinstance(choices, dict) or not choices:
        raise ValueError(f"row {row.get('id', '<unknown>')} has invalid question/choices")
    rendered_choices = "\n".join(
        f"{key}. {str(choices[key]).strip()}" for key in sorted(choices)
    )
    return f"{question}\n\n选项：\n{rendered_choices}"


def dedup_key(row: dict[str, Any]) -> str:
    return normalize_for_dedup(render_question(row))


def render_training_document(row: dict[str, Any]) -> str:
    answer = str(row.get("reference_answer") or "").strip().upper()
    if not answer:
        raise ValueError(f"row {row.get('id', '<unknown>')} has no reference answer")
    return f"{render_question(row)}\n\n参考答案：{answer}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_corpus(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    val_keys = {dedup_key(row) for row in val_rows}
    seen_train: set[str] = set()
    documents: list[dict[str, Any]] = []
    duplicate_train_rows = 0
    overlap_train_rows = 0

    for row in train_rows:
        key = dedup_key(row)
        if key in seen_train:
            duplicate_train_rows += 1
            continue
        seen_train.add(key)
        if key in val_keys:
            overlap_train_rows += 1
            continue
        index = len(documents)
        documents.append(
            {
                "doc_id": f"cmb-train-{index:05d}",
                "title": f"CMB-Exam 训练例题 {index:05d}",
                "text": render_training_document(row),
                "source": "FreedomIntelligence/CMB::CMB-Exam::train",
                "license": "Apache-2.0 (declared; re-check provenance)",
                "metadata": {
                    "source_id": str(row.get("id") or ""),
                    "source_split": "train",
                    "exam_class": str(row.get("exam_class") or ""),
                    "exam_subject": str(row.get("exam_subject") or ""),
                    "reference_answer": str(row.get("reference_answer") or "").upper(),
                    "retrieval_role": "solved-training-example",
                },
            }
        )

    benchmark = []
    for row in val_rows:
        benchmark.append(
            {
                "id": str(row.get("id") or ""),
                "question": render_question(row),
                "choices": row.get("choices") or {},
                "reference_answer": str(row.get("reference_answer") or "").upper(),
                "task_type": str(row.get("task_type") or "single_choice"),
                "source_split": "val",
                "relevant_doc_ids": [],
                "retrieval_labels_available": False,
            }
        )

    metadata = {
        "task": "closed_domain_cmb_exam_rag",
        "corpus_role": "retrieved-solved-training-examples",
        "declared_license": "Apache-2.0",
        "source_manifest": "data/sources/cmb-exam.yaml",
        "raw_data_committed_to_github": False,
        "answers_in_retrieval_text": True,
        "explanations_in_retrieval_text": False,
        "train_rows_input": len(train_rows),
        "val_rows_input": len(val_rows),
        "train_duplicate_rows_removed": duplicate_train_rows,
        "train_cross_split_overlap_rows_removed": overlap_train_rows,
        "corpus_documents": len(documents),
        "val_queries": len(benchmark),
        "val_unique_queries": len({dedup_key(row) for row in val_rows}),
        "retrieval_labels_available": False,
    }
    return documents, benchmark, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train_path = args.input_dir / "train.jsonl"
    val_path = args.input_dir / "val.jsonl"
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    documents, benchmark, metadata = prepare_corpus(train_rows, val_rows)
    metadata.update(
        {
            "train_input": str(train_path),
            "val_input": str(val_path),
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
        }
    )
    write_jsonl(args.output_dir / "knowledge_base.jsonl", documents)
    write_jsonl(args.output_dir / "qa_benchmark.jsonl", benchmark)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
