from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.evaluation import extract_choice, score_multiple_choice


class EvaluationTest(unittest.TestCase):
    choices = {"A": "first", "B": "second", "C": "third"}

    def test_extracts_labeled_choice(self) -> None:
        self.assertEqual(extract_choice("答案：B", self.choices), "B")

    def test_extracts_leading_choice(self) -> None:
        self.assertEqual(extract_choice("A. first", self.choices), "A")

    def test_ignores_invalid_choice(self) -> None:
        self.assertIsNone(extract_choice("答案：D", self.choices))

    def test_scores_accuracy_and_invalid_rate(self) -> None:
        rows = [
            {"id": "1", "choices": self.choices, "reference_answer": "A", "answer": "A"},
            {"id": "2", "choices": self.choices, "reference_answer": "B", "answer": "答案：C"},
            {"id": "3", "choices": self.choices, "reference_answer": "C", "answer": "无法判断"},
        ]
        report = score_multiple_choice(rows)
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["accuracy"], 0.3333)
        self.assertEqual(report["invalid_rate"], 0.3333)


if __name__ == "__main__":
    unittest.main()

