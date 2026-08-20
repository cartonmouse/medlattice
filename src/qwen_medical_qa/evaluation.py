"""Small, transparent evaluators for the project's fixed benchmarks."""

from __future__ import annotations

import re
from typing import Any, Iterable


_LABELED_CHOICE = re.compile(
    r"(?:答案|选项|选择|answer|option)\s*(?:是|为|为：|是：|[:：])?\s*[\[【(（]?([A-Z])[\]】)）]?")
_LEADING_CHOICE = re.compile(r"^\s*[\[【(（]?([A-Z])[\]】)）.。:：]?(?:\s|$)")
_STANDALONE_CHOICE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def extract_choice(answer: str, valid_choices: Iterable[str]) -> str | None:
    """Extract the first valid option letter from a model answer."""
    valid = {str(choice).strip().upper() for choice in valid_choices}
    if not valid:
        return None

    text = answer.strip().upper()
    for pattern in (_LABELED_CHOICE, _LEADING_CHOICE, _STANDALONE_CHOICE):
        match = pattern.search(text)
        if match and match.group(1) in valid:
            return match.group(1)
    return None


def score_multiple_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score generated answers using exact option-letter matching."""
    details: list[dict[str, Any]] = []
    correct = 0
    invalid = 0

    for row in rows:
        choices = row.get("choices") or {}
        reference = str(row.get("reference_answer", "")).strip().upper()
        predicted = extract_choice(row.get("answer", ""), choices.keys())
        is_correct = predicted is not None and predicted == reference
        if is_correct:
            correct += 1
        if predicted is None:
            invalid += 1
        details.append(
            {
                "id": row.get("id"),
                "predicted": predicted,
                "reference": reference,
                "correct": is_correct,
                "answer": row.get("answer", ""),
            }
        )

    total = len(rows)
    return {
        "task": "multiple_choice_exact_letter",
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": round(correct / total, 4) if total else None,
        "invalid_rate": round(invalid / total, 4) if total else None,
        "details": details,
    }

