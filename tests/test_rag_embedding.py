from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import Chunk  # noqa: E402
from qwen_medical_qa.rag_embedding import DenseRetriever  # noqa: E402


class DenseRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("doc-a#chunk-000", "doc-a", "A", "A"),
            Chunk("doc-b#chunk-000", "doc-b", "B", "B"),
            Chunk("doc-c#chunk-000", "doc-c", "C", "C"),
        ]
        self.retriever = DenseRetriever(
            self.chunks,
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            model_name="test-model",
        )

    def test_ranks_by_cosine_similarity(self) -> None:
        results = self.retriever.search([0.9, 0.1], top_k=2)
        self.assertEqual([result.chunk.doc_id for result in results], ["doc-a", "doc-b"])
        self.assertGreater(results[0].score, results[1].score)

    def test_min_score_filters_results(self) -> None:
        results = self.retriever.search([1.0, 0.0], top_k=3, min_score=0.5)
        self.assertEqual([result.chunk.doc_id for result in results], ["doc-a"])

    def test_index_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense-index.json"
            self.retriever.save(path)
            loaded = DenseRetriever.load(path)
            self.assertEqual(loaded.model_name, "test-model")
            self.assertEqual(loaded.dimension, 2)
            self.assertEqual(loaded.search([0.0, 1.0], top_k=1)[0].chunk.doc_id, "doc-b")


if __name__ == "__main__":
    unittest.main()
