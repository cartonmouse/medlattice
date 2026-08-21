from __future__ import annotations

import unittest

from qwen_medical_qa.grpo_data import (
    build_grpo_record,
    choice_exact_match_reward,
    choice_format_reward,
    completion_text,
    split_grpo_records,
)


def make_row(row_id: str = "row-1", answer: str = "A") -> dict[str, object]:
    return {
        "id": row_id,
        "question": "Which option is correct?",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma"},
        "reference_answer": answer,
        "task_type": "multiple_choice",
        "source": "test",
        "source_split": "train",
    }


class GRPODataTests(unittest.TestCase):
    def test_record_keeps_dynamic_option_keys_and_conversation_prompt(self) -> None:
        row = make_row()
        row["choices"] = {"A": "alpha", "B": "beta", "F": "foxtrot"}
        record = build_grpo_record(row)
        self.assertEqual(record["valid_choices"], ["A", "B", "F"])
        self.assertEqual([message["role"] for message in record["prompt"]], ["system", "user"])

    def test_split_is_reproducible_complete_and_disjoint(self) -> None:
        records = [build_grpo_record(make_row(f"row-{index}")) for index in range(10)]
        train_a, eval_a = split_grpo_records(records, eval_size=3, seed=42)
        train_b, eval_b = split_grpo_records(records, eval_size=3, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual(len(train_a), 7)
        self.assertEqual(len(eval_a), 3)
        self.assertTrue(
            {row["id"] for row in train_a}.isdisjoint(row["id"] for row in eval_a)
        )

    def test_completion_text_supports_trl_conversation(self) -> None:
        completion = [{"role": "assistant", "content": "B"}]
        self.assertEqual(completion_text(completion), "B")
        self.assertEqual(completion_text(" C "), " C ")

    def test_exact_match_reward_uses_project_choice_parser(self) -> None:
        rewards = choice_exact_match_reward(
            completions=[
                [{"role": "assistant", "content": "A"}],
                [{"role": "assistant", "content": "答案是 B"}],
                [{"role": "assistant", "content": "无法确定"}],
            ],
            reference_answer=["A", "A", "A"],
            valid_choices=[["A", "B"], ["A", "B"], ["A", "B"]],
        )
        self.assertEqual(rewards, [1.0, 0.0, 0.0])

    def test_format_reward_requires_single_valid_letter(self) -> None:
        rewards = choice_format_reward(
            completions=["A", "答案是 A", "F"],
            valid_choices=[["A", "B"], ["A", "B"], ["A", "B"]],
        )
        self.assertEqual(rewards, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
