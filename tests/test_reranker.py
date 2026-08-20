import unittest

from qwen_medical_qa.rag import Chunk, RetrievalResult
from qwen_medical_qa.reranker import BM25Reranker


class BM25RerankerTests(unittest.TestCase):
    def test_reranks_candidate_by_query_terms(self) -> None:
        candidates = [
            RetrievalResult(
                rank=1,
                score=0.9,
                chunk=Chunk(
                    chunk_id="objective#chunk-000",
                    doc_id="objective",
                    title="Objective sign",
                    text="an objective finding from examination",
                ),
            ),
            RetrievalResult(
                rank=2,
                score=0.4,
                chunk=Chunk(
                    chunk_id="subjective#chunk-000",
                    doc_id="subjective",
                    title="Subjective symptom",
                    text="a subjective symptom reported by the patient",
                ),
            ),
        ]
        results = BM25Reranker().rerank("subjective symptom", candidates, top_k=2)
        self.assertEqual(results[0].chunk.doc_id, "subjective")
        self.assertAlmostEqual(results[0].retrieval_score, 0.4)

    def test_empty_candidates_and_invalid_parameters(self) -> None:
        self.assertEqual(BM25Reranker().rerank("query", []), [])
        with self.assertRaises(ValueError):
            BM25Reranker(k1=0)
        with self.assertRaises(ValueError):
            BM25Reranker(b=2)
