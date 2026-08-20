import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.retrieval_compare import compare_retrieval_rows


class RetrievalCompareTests(unittest.TestCase):
    def test_overlap_statistics(self) -> None:
        left = [
            {
                "id": "q1",
                "retriever": "sqlite-dense",
                "retrieved": [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}],
            }
        ]
        right = [
            {
                "id": "q1",
                "retriever": "faiss-hnsw",
                "retrieved": [{"chunk_id": "a"}, {"chunk_id": "c"}, {"chunk_id": "b"}],
            }
        ]
        result = compare_retrieval_rows(left, right, top_k=3)
        self.assertEqual(result["exact_ranking_match"], 0)
        self.assertEqual(result["exact_set_match"], 1)
        self.assertEqual(result["mean_set_overlap"], 1.0)

    def test_query_ids_must_match(self) -> None:
        with self.assertRaises(ValueError):
            compare_retrieval_rows(
                [{"id": "left", "retrieved": []}],
                [{"id": "right", "retrieved": []}],
            )


if __name__ == "__main__":
    unittest.main()
