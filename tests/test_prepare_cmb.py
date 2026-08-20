from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from prepare_cmb import (
    deterministic_limit,
    deterministic_reservoir,
    has_valid_reference_answer,
    is_single_choice,
    normalize_record,
)


class PrepareCMBTest(unittest.TestCase):
    raw = {
        "id": "demo-1",
        "exam_type": "demo",
        "exam_subject": "基础医学",
        "question": "下列哪项是示例答案？",
        "question_type": "单项选择题",
        "option": {"A": "错误", "B": "正确"},
        "answer": "B",
        "explanation": "仅用于测试格式。",
    }

    def test_normalizes_cmb_shape(self) -> None:
        row = normalize_record({**self.raw, "option": json.dumps(self.raw["option"])}, "train", 0)
        self.assertEqual(row["id"], "cmb-exam-train-demo-1")
        self.assertEqual(row["reference_answer"], "B")
        self.assertTrue(is_single_choice(row))

    def test_allows_unlabeled_test_record(self) -> None:
        raw = {key: value for key, value in self.raw.items() if key not in {"answer", "explanation"}}
        row = normalize_record(raw, "test", 0)
        self.assertIsNone(row["reference_answer"])
        self.assertTrue(is_single_choice(row))

    def test_rejects_multiple_answer_for_single_choice_filter(self) -> None:
        row = normalize_record({**self.raw, "answer": "AB", "question_type": "多项选择题"}, "train", 0)
        self.assertFalse(is_single_choice(row))

    def test_rejects_answer_missing_from_choices(self) -> None:
        row = normalize_record({**self.raw, "answer": "E"}, "train", 0)
        self.assertFalse(has_valid_reference_answer(row))

    def test_deterministic_limit(self) -> None:
        rows = [{"id": str(index)} for index in range(20)]
        first = deterministic_limit(rows, 5, 42)
        second = deterministic_limit(rows, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_deterministic_reservoir(self) -> None:
        rows = [{"id": str(index)} for index in range(20)]
        first = deterministic_reservoir(rows, 5, 42)
        second = deterministic_reservoir(rows, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)


if __name__ == "__main__":
    unittest.main()
