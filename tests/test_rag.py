from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import (  # noqa: E402
    Document,
    TfidfRetriever,
    build_rag_prompt,
    chunk_document,
    tokenize,
)


class RagTest(unittest.TestCase):
    def setUp(self) -> None:
        documents = [
            Document("symptom", "症状", "症状是患者主观感受到的异常表现。", "synthetic"),
            Document("sign", "体征", "体征是通过检查发现的客观表现。", "synthetic"),
        ]
        chunks = [chunk for document in documents for chunk in chunk_document(document, 100, 10)]
        self.retriever = TfidfRetriever(chunks)

    def test_tokenizes_chinese_characters_and_english_words(self) -> None:
        self.assertEqual(tokenize("症状 Qwen3"), ["症", "状", "qwen3"])

    def test_chunk_overlap_is_deterministic(self) -> None:
        document = Document("doc", "title", "abcdefghij")
        chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)
        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "defg", "ghij"])
        self.assertEqual([chunk.start for chunk in chunks], [0, 3, 6])

    def test_retrieves_matching_document(self) -> None:
        results = self.retriever.search("患者主观感受到的异常表现", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.doc_id, "symptom")
        self.assertGreater(results[0].score, 0)

    def test_unknown_query_returns_no_false_context(self) -> None:
        self.assertEqual(self.retriever.search("火星量子信号", top_k=3), [])

    def test_index_round_trip_and_prompt_citation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.retriever.save(path)
            loaded = TfidfRetriever.load(path)
            results = loaded.search("客观表现", top_k=1)
            prompt = build_rag_prompt("什么是体征？", results)
            self.assertEqual(results[0].chunk.doc_id, "sign")
            self.assertIn("[sign#chunk-000]", prompt)
            self.assertIn("资料不足，无法判断", prompt)


if __name__ == "__main__":
    unittest.main()
