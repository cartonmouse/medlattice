from __future__ import annotations

import unittest

from qwen_medical_qa.hard_negative_data import (
    build_hard_negative_record,
    choose_hard_negative,
)


def make_row() -> dict[str, object]:
    return {
        "id": "cmb-train-1",
        "question": "Which option is correct?",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma"},
        "reference_answer": "A",
        "task_type": "multiple_choice",
        "source": "FreedomIntelligence/CMB",
        "source_split": "train",
    }


class HardNegativeDataTests(unittest.TestCase):
    def test_selects_highest_scoring_wrong_option(self) -> None:
        model_top, hard_negative = choose_hard_negative(
            make_row(), {"A": -0.8, "B": -0.2, "C": -0.4}
        )
        self.assertEqual(model_top, "B")
        self.assertEqual(hard_negative, "B")

    def test_record_contains_model_confusion_metadata(self) -> None:
        record = build_hard_negative_record(
            make_row(),
            {"A": -0.1, "B": -0.4, "C": -0.7},
            generator_model="Qwen/Qwen3-1.7B",
            generator_adapter="outputs/qlora-cmb-v1",
        )
        self.assertEqual(record["chosen"][0]["content"], "A")
        self.assertEqual(record["rejected"][0]["content"], "B")
        self.assertTrue(record["model_prediction_correct"])
        self.assertEqual(record["negative_strategy"], "highest_scoring_wrong_option")

    def test_rejects_missing_option_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing option scores"):
            choose_hard_negative(make_row(), {"A": -0.1, "B": -0.2})


if __name__ == "__main__":
    unittest.main()
