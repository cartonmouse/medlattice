import random
import unittest

from qwen_medical_qa.dpo_data import (
    build_preference_pairs,
    build_preference_record,
    split_preference_pairs,
)
from qwen_medical_qa.dpo_training import render_preference_rows


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


class DPODataTests(unittest.TestCase):
    def test_record_uses_same_prompt_and_distinct_completions(self) -> None:
        record = build_preference_record(make_row(), random.Random(7))
        self.assertEqual([message["role"] for message in record["prompt"]], ["system", "user"])
        self.assertEqual(record["chosen"][0], {"role": "assistant", "content": "A"})
        self.assertNotEqual(record["chosen"], record["rejected"])
        self.assertNotEqual(record["rejected_answer"], "A")

    def test_build_is_reproducible(self) -> None:
        rows = [make_row(f"row-{index}") for index in range(10)]
        self.assertEqual(build_preference_pairs(rows, seed=42), build_preference_pairs(rows, seed=42))
        self.assertNotEqual(build_preference_pairs(rows, seed=42), build_preference_pairs(rows, seed=43))

    def test_split_is_complete_and_disjoint(self) -> None:
        rows = [make_row(f"row-{index}") for index in range(10)]
        pairs = build_preference_pairs(rows, seed=42)
        train, evaluation = split_preference_pairs(pairs, eval_size=3, seed=42)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(evaluation), 3)
        self.assertEqual(
            {row["id"] for row in train} | {row["id"] for row in evaluation},
            {row["id"] for row in pairs},
        )
        self.assertTrue({row["id"] for row in train}.isdisjoint(row["id"] for row in evaluation))

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate preference id"):
            build_preference_pairs([make_row(), make_row()], seed=42)

    def test_rejects_invalid_eval_size(self) -> None:
        pairs = build_preference_pairs([make_row(f"row-{index}") for index in range(3)], seed=42)
        with self.assertRaises(ValueError):
            split_preference_pairs(pairs, eval_size=3, seed=42)

    def test_rendering_produces_standard_trl_fields(self) -> None:
        record = build_preference_record(make_row(), random.Random(7))

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered prompt"

        tokenizer = FakeTokenizer()
        rendered = render_preference_rows([record], tokenizer)
        self.assertEqual(
            rendered[0],
            {
                "prompt": "rendered prompt",
                "chosen": "A",
                "rejected": record["rejected_answer"],
            },
        )
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_rendering_supports_legacy_chat_template_signature(self) -> None:
        record = build_preference_record(make_row(), random.Random(7))

        class LegacyTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "legacy prompt"

        rendered = render_preference_rows([record], LegacyTokenizer())
        self.assertEqual(rendered[0]["prompt"], "legacy prompt")


if __name__ == "__main__":
    unittest.main()
