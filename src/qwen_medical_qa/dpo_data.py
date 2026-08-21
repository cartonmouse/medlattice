"""Build and validate conversational preference pairs for DPO."""

from __future__ import annotations

import random
from typing import Any, Iterable

from qwen_medical_qa.prompting import build_messages


PREFERENCE_SOURCE = "synthetic_from_reference_answer"
NEGATIVE_STRATEGY = "random_wrong_choice"


def _choice_keys(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    if not isinstance(choices, dict):
        raise ValueError(f"{row.get('id', '<unknown>')} choices must be an object")
    keys = sorted(str(key) for key in choices)
    if len(keys) < 2:
        raise ValueError(f"{row.get('id', '<unknown>')} needs at least two choices")
    return keys


def choose_rejected_answer(row: dict[str, Any], rng: random.Random) -> str:
    """Choose a deterministic wrong option from a labeled single-choice row."""

    answer = row.get("reference_answer")
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"{row.get('id', '<unknown>')} has no reference answer")
    keys = _choice_keys(row)
    if answer not in keys:
        raise ValueError(f"{row.get('id', '<unknown>')} reference answer is not a choice")
    wrong_choices = [key for key in keys if key != answer]
    return rng.choice(wrong_choices)


def build_preference_record(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Convert one normalized CMB row into TRL's explicit conversational format."""

    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("preference rows require a non-empty id")
    answer = row.get("reference_answer")
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"{row_id} has no reference answer")
    rejected_answer = choose_rejected_answer(row, rng)
    prompt = build_messages(row)
    record = {
        "id": row_id,
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": answer}],
        "rejected": [{"role": "assistant", "content": rejected_answer}],
        "reference_answer": answer,
        "rejected_answer": rejected_answer,
        "preference_source": PREFERENCE_SOURCE,
        "negative_strategy": NEGATIVE_STRATEGY,
        "task_type": row.get("task_type", "multiple_choice"),
        "source": row.get("source"),
        "source_split": row.get("source_split"),
    }
    validate_preference_record(record)
    return record


def validate_preference_record(record: dict[str, Any]) -> None:
    """Raise ValueError when a record is not a valid explicit preference pair."""

    row_id = record.get("id", "<unknown>")
    prompt = record.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"{row_id} prompt must contain system and user messages")
    if [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"{row_id} prompt must be system then user")
    for message in prompt:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{row_id} prompt messages must contain text")

    completions: list[str] = []
    for field in ("chosen", "rejected"):
        value = record.get(field)
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError(f"{row_id} {field} must contain one completion message")
        message = value[0]
        if message.get("role") != "assistant":
            raise ValueError(f"{row_id} {field} must be an assistant message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{row_id} {field} must contain text")
        completions.append(content)
    if completions[0] == completions[1]:
        raise ValueError(f"{row_id} chosen and rejected must differ")


def build_preference_pairs(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Build pairs in input order with reproducible wrong-option sampling."""

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        pair = build_preference_record(row, rng)
        if pair["id"] in seen_ids:
            raise ValueError(f"duplicate preference id: {pair['id']}")
        seen_ids.add(pair["id"])
        pairs.append(pair)
    if not pairs:
        raise ValueError("no preference pairs were built")
    return pairs


def split_preference_pairs(
    pairs: list[dict[str, Any]], eval_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make a deterministic preference holdout without touching the final CMB val set."""

    if eval_size < 0 or eval_size >= len(pairs):
        raise ValueError(f"eval_size must be in [0, {len(pairs) - 1}]")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    eval_indices = set(indices[:eval_size])
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in eval_indices]
    eval_pairs = [pair for index, pair in enumerate(pairs) if index in eval_indices]
    return train_pairs, eval_pairs
