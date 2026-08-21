"""Adapters and audit helpers for public medical preference datasets."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from qwen_medical_qa.dpo_data import validate_preference_record
from qwen_medical_qa.prompting import SYSTEM_PROMPT


PUBLIC_DATASET_ID = "TsinghuaC3I/UltraMedical-Preference"
PUBLIC_DATASET_REVISION = "761eb7935310ba662a96d93c5af342e5269d5759"
PUBLIC_PREFERENCE_SOURCE = "public_ultramedical_preference"


@dataclass
class PublicFilterStats:
    """Counts retained while turning a raw public split into preference pairs."""

    input_rows: int = 0
    accepted_rows: int = 0
    skipped_label: int = 0
    skipped_missing_score: int = 0
    skipped_non_strict_score: int = 0
    skipped_duplicate_prompt_id: int = 0
    skipped_duplicate_pair: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _prompt_id(row: dict[str, Any]) -> str:
    value = row.get("prompt_id")
    if value is not None and str(value).strip():
        return str(value).strip()
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("public preference rows require prompt or prompt_id")
    return hashlib.sha256(_normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def prompt_fingerprint(prompt: str) -> str:
    """Return a stable hash for overlap checks without storing prompt text in reports."""

    return hashlib.sha256(_normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _assistant_content(value: Any, row_id: str, field: str) -> str:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{row_id} {field} must contain messages")
    assistant_parts: list[str] = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"{row_id} {field} contains a non-object message")
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{row_id} {field} contains an empty assistant message")
        assistant_parts.append(content.strip())
    if not assistant_parts:
        raise ValueError(f"{row_id} {field} must contain an assistant message")
    return "\n\n".join(assistant_parts)


def _side_metadata(row: dict[str, Any], side: str) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    side_metadata = metadata.get(side)
    if not isinstance(side_metadata, dict):
        return {}
    keep = ("evaluation", "model", "rank", "score")
    return {key: side_metadata[key] for key in keep if key in side_metadata}


def preference_score(row: dict[str, Any], side: str) -> float | None:
    """Read a public record's chosen/rejected model score when one is available."""

    value = _side_metadata(row, side).get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_public_preference_record(
    row: dict[str, Any],
    *,
    source_file: str,
    source_split: str,
) -> dict[str, Any]:
    """Convert one UltraMedical row to the project's explicit conversational format."""

    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{_prompt_id(row)} prompt must be non-empty text")
    original_id = _prompt_id(row)
    chosen = _assistant_content(row.get("chosen"), original_id, "chosen")
    rejected = _assistant_content(row.get("rejected"), original_id, "rejected")
    if chosen == rejected:
        raise ValueError(f"{original_id} chosen and rejected must differ")

    record = {
        "id": f"ultramedical-{original_id}",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "preference_source": PUBLIC_PREFERENCE_SOURCE,
        "dataset_id": PUBLIC_DATASET_ID,
        "dataset_revision": PUBLIC_DATASET_REVISION,
        "source_file": source_file,
        "source_split": source_split,
        "prompt_id": original_id,
        "label_type": str(row.get("label_type", "unknown")),
        "chosen_score": preference_score(row, "chosen"),
        "rejected_score": preference_score(row, "rejected"),
        "source_metadata": {
            "chosen": _side_metadata(row, "chosen"),
            "rejected": _side_metadata(row, "rejected"),
        },
    }
    validate_preference_record(record)
    return record


def build_public_preference_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    source_file: str,
    source_split: str,
    require_strict_score: bool = True,
    allowed_label_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], PublicFilterStats]:
    """Filter and convert public rows with auditable, deterministic rules."""

    pairs: list[dict[str, Any]] = []
    stats = PublicFilterStats()
    seen_prompt_ids: set[str] = set()
    seen_pair_keys: set[tuple[str, str, str]] = set()

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("public preference split must contain JSON objects")
        stats.input_rows += 1
        label_type = str(row.get("label_type", "unknown"))
        if allowed_label_types is not None and label_type not in allowed_label_types:
            stats.skipped_label += 1
            continue

        chosen_score = preference_score(row, "chosen")
        rejected_score = preference_score(row, "rejected")
        if require_strict_score and (chosen_score is None or rejected_score is None):
            stats.skipped_missing_score += 1
            continue
        if (
            require_strict_score
            and chosen_score is not None
            and rejected_score is not None
            and chosen_score <= rejected_score
        ):
            stats.skipped_non_strict_score += 1
            continue

        pair = build_public_preference_record(
            row,
            source_file=source_file,
            source_split=source_split,
        )
        prompt_id = str(pair["prompt_id"])
        pair_key = (
            prompt_fingerprint(pair["prompt"][1]["content"]),
            pair["chosen"][0]["content"],
            pair["rejected"][0]["content"],
        )
        if prompt_id in seen_prompt_ids:
            stats.skipped_duplicate_prompt_id += 1
            continue
        if pair_key in seen_pair_keys:
            stats.skipped_duplicate_pair += 1
            continue
        seen_prompt_ids.add(prompt_id)
        seen_pair_keys.add(pair_key)
        pairs.append(pair)
        stats.accepted_rows += 1

    if not pairs:
        raise ValueError("no public preference pairs remain after filtering")
    return pairs, stats


def select_preference_pairs(
    pairs: list[dict[str, Any]], max_pairs: int | None, seed: int
) -> list[dict[str, Any]]:
    """Select a reproducible subset while retaining source order in the output."""

    if max_pairs is None or max_pairs >= len(pairs):
        return list(pairs)
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    import random

    selected_indices = sorted(random.Random(seed).sample(range(len(pairs)), max_pairs))
    return [pairs[index] for index in selected_indices]


def prompt_overlap(
    left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]]
) -> set[str]:
    """Return normalized prompt hashes shared by two converted preference splits."""

    left_hashes = {
        prompt_fingerprint(row["prompt"][1]["content"])
        for row in left
    }
    right_hashes = {
        prompt_fingerprint(row["prompt"][1]["content"])
        for row in right
    }
    return left_hashes & right_hashes
