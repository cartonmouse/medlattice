import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.prediction_compare import compare_prediction_rows
from qwen_medical_qa.reproducibility import configure_reproducibility


class ReproducibilityTests(unittest.TestCase):
    def test_seed_is_recorded_and_reseeds_python(self) -> None:
        first = configure_reproducibility(7)
        first_random = random.random()
        second = configure_reproducibility(7)
        second_random = random.random()
        self.assertEqual(first["seed"], 7)
        self.assertEqual(first_random, second_random)
        self.assertEqual(second["seed"], 7)

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            configure_reproducibility(-1)

    def test_prediction_drift_matrix(self) -> None:
        left = [
            {"id": "1", "answer": "A", "reference_answer": "A", "choices": {"A": ""}},
            {"id": "2", "answer": "B", "reference_answer": "A", "choices": {"A": "", "B": ""}},
            {"id": "3", "answer": "C", "reference_answer": "C", "choices": {"A": "", "C": ""}},
        ]
        right = [
            {"id": "1", "answer": "A", "reference_answer": "A", "choices": {"A": ""}},
            {"id": "2", "answer": "A", "reference_answer": "A", "choices": {"A": "", "B": ""}},
            {"id": "3", "answer": "D", "reference_answer": "C", "choices": {"A": "", "C": "", "D": ""}},
        ]
        result = compare_prediction_rows(left, right)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["same_prediction"], 1)
        self.assertEqual(result["left_correct_right_wrong"], 1)
        self.assertEqual(result["left_wrong_right_correct"], 1)


if __name__ == "__main__":
    unittest.main()
