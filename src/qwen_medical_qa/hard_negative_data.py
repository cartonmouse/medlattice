"""Build CMB hard-negative preference pairs from option log-probabilities."""

from __future__ import annotations

from typing import Any, Mapping

from qwen_medical_qa.dpo_data import validate_preference_record
from qwen_medical_qa.prompting import build_messages


PREFERENCE_SOURCE = "sft_hard_negative_from_option_logprob"
NEGATIVE_STRATEGY = "highest_scoring_wrong_option"


def option_keys(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    if not isinstance(choices, dict):
        raise ValueError(f"{row.get('id', '<unknown>')} choices must be an object")
    keys = sorted(str(key).strip().upper() for key in choices)
    if len(keys) < 2:
        raise ValueError(f"{row.get('id', '<unknown>')} needs at least two choices")
    return keys


def choose_hard_negative(
    row: dict[str, Any], option_scores: Mapping[str, float]
) -> tuple[str, str]:
    """Return model top choice and highest-scoring wrong choice."""

    row_id = str(row.get("id", "<unknown>"))
    reference = row.get("reference_answer")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{row_id} has no reference answer")
    reference = reference.strip().upper()
    keys = option_keys(row)
    if reference not in keys:
        raise ValueError(f"{row_id} reference answer is not a choice")
    missing = [key for key in keys if key not in option_scores]
    if missing:
        raise ValueError(f"{row_id} missing option scores: {missing}")

    model_top = max(keys, key=lambda key: (float(option_scores[key]), key))
    wrong_keys = [key for key in keys if key != reference]
    hard_negative = max(wrong_keys, key=lambda key: (float(option_scores[key]), key))
    return model_top, hard_negative


def build_hard_negative_record(
    row: dict[str, Any],
    option_scores: Mapping[str, float],
    *,
    generator_model: str,
    generator_adapter: str,
) -> dict[str, Any]:
    """Convert one normalized CMB row into an explicit hard-negative DPO pair."""

    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("hard-negative rows require a non-empty id")
    reference = row.get("reference_answer")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{row_id} has no reference answer")
    reference = reference.strip().upper()
    model_top, hard_negative = choose_hard_negative(row, option_scores)
    record = {
        "id": row_id,
        "prompt": build_messages(row),
        "chosen": [{"role": "assistant", "content": reference}],
        "rejected": [{"role": "assistant", "content": hard_negative}],
        "reference_answer": reference,
        "rejected_answer": hard_negative,
        "model_top_choice": model_top,
        "model_prediction_correct": model_top == reference,
        "reference_option_logprob": float(option_scores[reference]),
        "hard_negative_option_logprob": float(option_scores[hard_negative]),
        "reference_minus_hard_negative": float(
            option_scores[reference] - option_scores[hard_negative]
        ),
        "option_logprobs": {key: float(option_scores[key]) for key in sorted(option_scores)},
        "preference_source": PREFERENCE_SOURCE,
        "negative_strategy": NEGATIVE_STRATEGY,
        "generator_model": generator_model,
        "generator_adapter": generator_adapter,
        "task_type": row.get("task_type", "multiple_choice"),
        "source": row.get("source"),
        "source_split": row.get("source_split"),
    }
    validate_preference_record(record)
    return record
