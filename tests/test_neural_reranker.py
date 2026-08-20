import unittest

from qwen_medical_qa.neural_reranker import (
    TransformerCrossEncoderReranker,
    format_passage,
)
from qwen_medical_qa.rag import Chunk, RetrievalResult


class NeuralRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            RetrievalResult(
                rank=1,
                score=0.9,
                chunk=Chunk(
                    chunk_id="a#chunk-000",
                    doc_id="a",
                    title="低 dense 分数",
                    text="候选 A",
                ),
            ),
            RetrievalResult(
                rank=2,
                score=0.4,
                chunk=Chunk(
                    chunk_id="b#chunk-000",
                    doc_id="b",
                    title="高 neural 分数",
                    text="候选 B",
                ),
            ),
        ]

    def test_injected_scorer_reranks_and_preserves_dense_score(self) -> None:
        seen: dict[str, object] = {}

        def score_fn(query: str, passages: list[str]) -> list[float]:
            seen["query"] = query
            seen["passages"] = passages
            return [0.1, 0.9]

        reranker = TransformerCrossEncoderReranker(
            model_name="test-reranker",
            score_fn=score_fn,
        )
        results = reranker.rerank("  当前问题  ", self.candidates, top_k=1)

        self.assertEqual(results[0].chunk.doc_id, "b")
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertAlmostEqual(results[0].retrieval_score, 0.4)
        self.assertEqual(seen["query"], "当前问题")
        self.assertEqual(seen["passages"], ["低 dense 分数\n候选 A", "高 neural 分数\n候选 B"])

    def test_ties_are_deterministic_and_empty_is_safe(self) -> None:
        reranker = TransformerCrossEncoderReranker(
            score_fn=lambda _query, passages: [1.0 for _ in passages]
        )
        results = reranker.rerank("query", self.candidates)
        self.assertEqual([item.chunk.doc_id for item in results], ["a", "b"])
        self.assertEqual(reranker.rerank("query", []), [])

    def test_validates_configuration_and_score_count(self) -> None:
        with self.assertRaises(ValueError):
            TransformerCrossEncoderReranker(batch_size=0, score_fn=lambda _, __: [])
        with self.assertRaises(ValueError):
            TransformerCrossEncoderReranker(max_length=0, score_fn=lambda _, __: [])
        reranker = TransformerCrossEncoderReranker(score_fn=lambda _, __: [1.0])
        with self.assertRaises(ValueError):
            reranker.rerank("query", self.candidates)

    def test_format_passage_uses_text_when_title_is_empty(self) -> None:
        candidate = RetrievalResult(
            rank=1,
            score=0.1,
            chunk=Chunk("id", "doc", "", "正文"),
        )
        self.assertEqual(format_passage(candidate), "正文")


if __name__ == "__main__":
    unittest.main()
