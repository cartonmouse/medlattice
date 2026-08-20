import tempfile
import unittest
from pathlib import Path

from qwen_medical_qa.rag import Chunk
from qwen_medical_qa.vector_store import SqliteVectorStore


class SqliteVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk(
                chunk_id="doc-a#chunk-000",
                doc_id="doc-a",
                title="A",
                text="alpha",
                source="test",
                license="CC0",
                start=0,
                end=5,
                metadata={"section": "one"},
            ),
            Chunk(
                chunk_id="doc-b#chunk-000",
                doc_id="doc-b",
                title="B",
                text="beta",
                source="test",
                license="CC0",
                start=0,
                end=4,
            ),
        ]

    def test_build_and_search_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors.sqlite"
            store = SqliteVectorStore.build(
                path,
                self.chunks,
                [[1.0, 0.0], [0.0, 1.0]],
                model_name="test-model",
                query_instruction="query: ",
            )
            self.assertEqual(store.count, 2)
            self.assertEqual(store.dimension, 2)
            self.assertEqual(store.model_name, "test-model")
            results = store.search([0.9, 0.1], top_k=1)
            self.assertEqual([result.chunk.doc_id for result in results], ["doc-a"])
            self.assertEqual(results[0].chunk.metadata["section"], "one")

            loaded = SqliteVectorStore(path)
            self.assertEqual(loaded.search([0.1, 0.9], top_k=1)[0].chunk.doc_id, "doc-b")

    def test_dimension_and_output_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors.sqlite"
            SqliteVectorStore.build(path, self.chunks, [[1.0, 0.0], [0.0, 1.0]], "test")
            with self.assertRaises(ValueError):
                SqliteVectorStore(path).search([1.0, 0.0, 0.0])
            with self.assertRaises(FileExistsError):
                SqliteVectorStore.build(path, self.chunks, [[1.0, 0.0], [0.0, 1.0]], "test")
