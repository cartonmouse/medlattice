from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag_qa import extract_citation_ids, score_rag_answers  # noqa: E402


class RagQaTest(unittest.TestCase):
    def test_extracts_chunk_citations(self) -> None:
        answer = "症状。依据 [term-symptom#chunk-000]。"
        self.assertEqual(extract_citation_ids(answer), ["term-symptom#chunk-000"])

    def test_scores_grounded_answer(self) -> None:
        rows = [
            {
                "id": "1",
                "question": "是什么？",
                "expected_answer": "症状",
                "answer": "答案是症状，引用 [term-symptom#chunk-000]。",
                "relevant_doc_ids": ["term-symptom"],
                "retrieved": [
                    {
                        "rank": 1,
                        "doc_id": "term-symptom",
                        "chunk_id": "term-symptom#chunk-000",
                    }
                ],
            }
        ]
        report = score_rag_answers(rows, k=3)
        self.assertEqual(report["retrieval_hit_rate_at_k"], 1.0)
        self.assertEqual(report["answer_contains_expected_rate"], 1.0)
        self.assertEqual(report["valid_citation_rate"], 1.0)
        self.assertEqual(report["grounded_answer_rate"], 1.0)

    def test_invalid_citation_is_not_grounded(self) -> None:
        rows = [
            {
                "id": "1",
                "question": "是什么？",
                "expected_answer": "症状",
                "answer": "答案是症状，引用 [other#chunk-000]。",
                "relevant_doc_ids": ["term-symptom"],
                "retrieved": [
                    {
                        "rank": 1,
                        "doc_id": "term-symptom",
                        "chunk_id": "term-symptom#chunk-000",
                    }
                ],
            }
        ]
        report = score_rag_answers(rows, k=3)
        self.assertEqual(report["answer_contains_expected_rate"], 1.0)
        self.assertEqual(report["valid_citation_rate"], 0.0)
        self.assertEqual(report["grounded_answer_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
