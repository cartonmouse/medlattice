import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.faiss_store import FaissVectorStore
from qwen_medical_qa.rag import Chunk


@unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu is optional")
class FaissStoreTests(unittest.TestCase):
    def test_build_and_search(self) -> None:
        chunks = [
            Chunk(chunk_id="a", doc_id="a", title="A", text="a"),
            Chunk(chunk_id="b", doc_id="b", title="B", text="b"),
            Chunk(chunk_id="c", doc_id="c", title="C", text="c"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.faiss"
            store = FaissVectorStore.build(
                path=path,
                chunks=chunks,
                embeddings=[[1, 0], [0, 1], [0.7, 0.7]],
                model_name="test-model",
                hnsw_m=2,
                ef_construction=8,
                ef_search=8,
            )
            results = store.search([1, 0], top_k=2)
            self.assertEqual(store.count, 3)
            self.assertEqual([item.chunk.chunk_id for item in results], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
