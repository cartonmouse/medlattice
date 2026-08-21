"""Data records and deterministic rewards for the local CMB GRPO experiment."""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from qwen_medical_qa.evaluation import extract_choice
from qwen_medical_qa.prompting import build_messages


GRPO_SOURCE = "cmb_train_exact_match_reward"


def _normalise_choices(row: dict[str, Any]) -> dict[str, str]:
    row_id = str(row.get("id", "<unknown>"))
    choices = row.get("choices")
    if not isinstance(choices, dict) or len(choices) < 2:
        raise ValueError(f"{row_id} choices must contain at least two options")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in choices.items():
        key = str(raw_key).strip().upper()
        value = str(raw_value).strip()
        if len(key) != 1 or not key.isalpha():
            raise ValueError(f"{row_id} has an invalid option key: {raw_key!r}")
        if not value:
            raise ValueError(f"{row_id} has an empty option value for {key}")
        if key in normalized:
            raise ValueError(f"{row_id} has duplicate option key: {key}")
        normalized[key] = value
    return normalized


def build_grpo_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one normalized CMB row into a conversational GRPO record."""

    row_id = str(row.get("id", "")).strip()
    question = str(row.get("question", "")).strip()
    if not row_id:
        raise ValueError("GRPO records require a non-empty id")
    if not question:
        raise ValueError(f"{row_id} question must be non-empty")
    if row.get("task_type", "multiple_choice") != "multiple_choice":
        raise ValueError(f"{row_id} is not a single-choice record")

    choices = _normalise_choices(row)
    reference_answer = str(row.get("reference_answer", "")).strip().upper()
    if reference_answer not in choices:
        raise ValueError(f"{row_id} reference answer is not one of the option keys")

    normalized_row = dict(row)
    normalized_row["question"] = question
    normalized_row["choices"] = choices
    normalized_row["reference_answer"] = reference_answer
    record = {
        "id": row_id,
        "prompt": build_messages(normalized_row),
        "reference_answer": reference_answer,
        "valid_choices": list(choices),
        "task_type": "multiple_choice",
        "source": str(row.get("source", "unknown")),
        "source_split": str(row.get("source_split", "unknown")),
    }
    validate_grpo_record(record)
    return record


def validate_grpo_record(record: dict[str, Any]) -> None:
    """Validate the compact record consumed by ``GRPOTrainer``."""

    row_id = str(record.get("id", "<unknown>"))
    if not str(record.get("id", "")).strip():
        raise ValueError("GRPO record id must be non-empty")
    prompt = record.get("prompt")
    if not isinstance(prompt, list) or len(prompt) < 2:
        raise ValueError(f"{row_id} prompt must contain system and user messages")
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise ValueError(f"{row_id} prompt contains an invalid message")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{row_id} prompt message content must be non-empty")

    valid_choices = record.get("valid_choices")
    if not isinstance(valid_choices, list) or len(valid_choices) < 2:
        raise ValueError(f"{row_id} valid_choices must contain at least two options")
    normalized_choices = [str(choice).strip().upper() for choice in valid_choices]
    if len(set(normalized_choices)) != len(normalized_choices):
        raise ValueError(f"{row_id} valid_choices must be unique")
    reference_answer = str(record.get("reference_answer", "")).strip().upper()
    if reference_answer not in normalized_choices:
        raise ValueError(f"{row_id} reference answer is not a valid option")


def split_grpo_records(
    records: Sequence[dict[str, Any]], eval_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a reproducible, complete, and disjoint train/eval split."""

    if len(records) < 2:
        raise ValueError("at least two GRPO records are required")
    if eval_size <= 0 or eval_size >= len(records):
        raise ValueError("eval_size must be greater than zero and smaller than record count")

    ids = [str(record.get("id", "")) for record in records]
    if any(not row_id for row_id in ids):
        raise ValueError("GRPO records require non-empty ids")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate GRPO record id")
    for record in records:
        validate_grpo_record(record)

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    train = [records[index] for index in indices[eval_size:]]
    evaluation = [records[index] for index in indices[:eval_size]]
    if {record["id"] for record in train} & {record["id"] for record in evaluation}:
        raise AssertionError("GRPO train/eval split is not disjoint")
    if len(train) + len(evaluation) != len(records):
        raise AssertionError("GRPO split is not complete")
    return train, evaluation


def completion_text(completion: Any) -> str:
    """Extract generated text from TRL conversational or plain completion forms."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", completion.get("text", ""))
        return content if isinstance(content, str) else str(content)
    if isinstance(completion, (list, tuple)):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", item.get("text", ""))
                if content is not None:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return "" if completion is None else str(completion)


def _choice_batches(valid_choices: Any, batch_size: int) -> list[list[str]]:
    if isinstance(valid_choices, str):
        return [[valid_choices]] * batch_size
    if not isinstance(valid_choices, (list, tuple)):
        return [[str(valid_choices)]] * batch_size
    if valid_choices and all(isinstance(item, (list, tuple)) for item in valid_choices):
        if len(valid_choices) != batch_size:
            raise ValueError("valid_choices batch size does not match completions")
        return [[str(choice) for choice in item] for item in valid_choices]
    choices = [str(choice) for choice in valid_choices]
    return [choices] * batch_size


def _reference_batch(reference_answer: Any, batch_size: int) -> list[str]:
    if isinstance(reference_answer, str):
        return [reference_answer] * batch_size
    if isinstance(reference_answer, (list, tuple)) and len(reference_answer) == batch_size:
        return [str(answer) for answer in reference_answer]
    return [str(reference_answer)] * batch_size


def choice_exact_match_reward(
    completions: list[Any], reference_answer: Any, valid_choices: Any, **_: Any
) -> list[float]:
    """Return 1 when the project evaluator extracts the reference option."""

    choices_batch = _choice_batches(valid_choices, len(completions))
    references = _reference_batch(reference_answer, len(completions))
    rewards: list[float] = []
    for completion, reference, choices in zip(completions, references, choices_batch, strict=True):
        predicted = extract_choice(completion_text(completion), choices)
        rewards.append(float(predicted == str(reference).strip().upper()))
    return rewards


def choice_format_reward(completions: list[Any], valid_choices: Any, **_: Any) -> list[float]:
    """Return 1 when the model emits exactly one valid option letter."""

    choices_batch = _choice_batches(valid_choices, len(completions))
    rewards: list[float] = []
    for completion, choices in zip(completions, choices_batch, strict=True):
        text = completion_text(completion).strip().upper()
        rewards.append(float(text in {str(choice).strip().upper() for choice in choices}))
    return rewards
