from __future__ import annotations

import unittest

from qwen_medical_qa.public_dpo_data import (
    build_public_preference_pairs,
    build_public_preference_record,
    prompt_overlap,
    select_preference_pairs,
)


def make_row(
    prompt_id: str = "p-1",
    *,
    label_type: str = "hard",
    chosen_score: float = 0.9,
    rejected_score: float = 0.2,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "prompt": "What is the safest next step?",
        "chosen": [
            {"role": "user", "content": "What is the safest next step?"},
            {"role": "assistant", "content": "The chosen medical response."},
        ],
        "rejected": [
            {"role": "user", "content": "What is the safest next step?"},
            {"role": "assistant", "content": "The rejected medical response."},
        ],
        "label_type": label_type,
        "metadata": {
            "chosen": {"score": chosen_score, "rank": 1},
            "rejected": {"score": rejected_score, "rank": 2},
        },
    }


class PublicDPODataTests(unittest.TestCase):
    def test_record_uses_project_prompt_and_last_assistant_message(self) -> None:
        record = build_public_preference_record(
            make_row(), source_file="dev.json", source_split="dev"
        )
        self.assertEqual([message["role"] for message in record["prompt"]], ["system", "user"])
        self.assertEqual(record["chosen"][0]["content"], "The chosen medical response.")
        self.assertEqual(record["chosen_score"], 0.9)

    def test_strict_score_filter_and_prompt_deduplication(self) -> None:
        rows = [
            make_row("p-1"),
            make_row("p-1"),
            make_row("p-2", chosen_score=0.5, rejected_score=0.5),
            make_row("p-3", chosen_score=0.1, rejected_score=0.9),
        ]
        pairs, stats = build_public_preference_pairs(
            rows, source_file="dev.json", source_split="dev"
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(stats.skipped_duplicate_prompt_id, 1)
        self.assertEqual(stats.skipped_non_strict_score, 2)

    def test_label_filter_keeps_human_rows_without_score_filter(self) -> None:
        rows = [
            make_row("human-1", label_type="human", chosen_score=0.1, rejected_score=0.9),
            make_row("hard-1", label_type="hard"),
        ]
        pairs, stats = build_public_preference_pairs(
            rows,
            source_file="test.json",
            source_split="test_human",
            require_strict_score=False,
            allowed_label_types={"human"},
        )
        self.assertEqual([pair["label_type"] for pair in pairs], ["human"])
        self.assertEqual(stats.skipped_label, 1)

    def test_selection_is_reproducible(self) -> None:
        rows = []
        for index in range(10):
            row = make_row(f"p-{index}")
            row["prompt"] = f"Question {index}"
            rows.append(row)
        pairs, _ = build_public_preference_pairs(
            rows, source_file="dev.json", source_split="dev"
        )
        self.assertEqual(
            select_preference_pairs(pairs, max_pairs=4, seed=42),
            select_preference_pairs(pairs, max_pairs=4, seed=42),
        )
        self.assertNotEqual(
            select_preference_pairs(pairs, max_pairs=4, seed=42),
            select_preference_pairs(pairs, max_pairs=4, seed=43),
        )

    def test_prompt_overlap_uses_normalized_hashes(self) -> None:
        left = [
            build_public_preference_record(
                make_row("left"), source_file="dev.json", source_split="dev"
            )
        ]
        right_row = make_row("right")
        right_row["prompt"] = "  What is the safest next step?\n"
        right = [
            build_public_preference_record(
                right_row, source_file="test.json", source_split="test_human"
            )
        ]
        self.assertEqual(len(prompt_overlap(left, right)), 1)


if __name__ == "__main__":
    unittest.main()
