import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from prepare_cmb_rag import prepare_corpus


class PrepareCmbRagTests(unittest.TestCase):
    def _row(self, row_id: str, question: str, answer: str = "A") -> dict:
        return {
            "id": row_id,
            "question": question,
            "choices": {"A": "选项甲", "B": "选项乙"},
            "reference_answer": answer,
            "exam_subject": "test",
        }

    def test_removes_train_duplicates_and_cross_split_overlap(self) -> None:
        train = [
            self._row("train-1", "同一个问题"),
            self._row("train-1-duplicate", "同一个问题"),
            self._row("train-overlap", "验证问题"),
            self._row("train-2", "训练问题"),
        ]
        val = [self._row("val-1", "验证问题", "B")]
        documents, benchmark, metadata = prepare_corpus(train, val)
        self.assertEqual([item["doc_id"] for item in documents], ["cmb-train-00000", "cmb-train-00001"])
        self.assertEqual(metadata["train_duplicate_rows_removed"], 1)
        self.assertEqual(metadata["train_cross_split_overlap_rows_removed"], 1)
        self.assertEqual(benchmark[0]["retrieval_labels_available"], False)
        self.assertIn("参考答案：A", documents[0]["text"])
