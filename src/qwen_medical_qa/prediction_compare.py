"""Compare prediction JSONL files without exposing question text."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("predicted") or "").strip().upper()


def _reference(row: dict[str, Any]) -> str:
    return str(row.get("reference_answer") or row.get("reference") or "").strip().upper()


def _valid_letters(row: dict[str, Any]) -> set[str]:
    choices = row.get("choices")
    if isinstance(choices, dict) and choices:
        return {str(key).strip().upper() for key in choices}
    return set("ABCDE")


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{label} contains a row without id")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate id: {row_id}")
        indexed[row_id] = row
    if not indexed:
        raise ValueError(f"{label} contains no rows")
    return indexed


def compare_prediction_rows(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return aggregate drift statistics for two prediction collections."""

    left = _index(left_rows, "left")
    right = _index(right_rows, "right")
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            f"prediction ids differ: missing_left={missing_left}, "
            f"missing_right={missing_right}"
        )

    same = 0
    left_correct_right_wrong = 0
    left_wrong_right_correct = 0
    both_correct = 0
    both_wrong = 0
    left_invalid = 0
    right_invalid = 0
    for row_id in sorted(left):
        left_row = left[row_id]
        right_row = right[row_id]
        left_answer = _answer(left_row)
        right_answer = _answer(right_row)
        reference = _reference(left_row) or _reference(right_row)
        left_is_correct = bool(reference) and left_answer == reference
        right_is_correct = bool(reference) and right_answer == reference
        left_is_invalid = left_answer not in _valid_letters(left_row)
        right_is_invalid = right_answer not in _valid_letters(right_row)
        same += int(left_answer == right_answer)
        left_invalid += int(left_is_invalid)
        right_invalid += int(right_is_invalid)
        if left_is_correct and right_is_correct:
            both_correct += 1
        elif not left_is_correct and not right_is_correct:
            both_wrong += 1
        elif left_is_correct:
            left_correct_right_wrong += 1
        else:
            left_wrong_right_correct += 1

    total = len(left)
    return {
        "total": total,
        "same_prediction": same,
        "different_prediction": total - same,
        "left_invalid": left_invalid,
        "right_invalid": right_invalid,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_correct_right_wrong": left_correct_right_wrong,
        "left_wrong_right_correct": left_wrong_right_correct,
    }
