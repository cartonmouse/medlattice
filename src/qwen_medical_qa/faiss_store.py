"""Optional FAISS HNSW storage for approximate dense retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .rag import Chunk, RetrievalResult


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS support is optional; install requirements-rag-ann.txt first"
        ) from exc
    return faiss


def _normalise_matrix(embeddings: Iterable[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(list(embeddings), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors cannot have zero norm")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class FaissVectorStore:
    """Persistent FAISS HNSW index with JSON sidecar metadata.

    The index uses inner product over L2-normalized vectors, equivalent to
    cosine similarity. Chunk metadata is kept in a sidecar JSON file so the
    binary FAISS index remains responsible only for vector search.
    """

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.metadata_path = Path(f"{self.path}.meta.json")
        if not self.path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"missing FAISS index or metadata: {self.path}")
        faiss = _require_faiss()
        self._index = faiss.read_index(str(self.path))
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if int(self._metadata.get("version", 0)) != self.VERSION:
            raise ValueError(
                f"unsupported FAISS index version: {self._metadata.get('version')}"
            )
        self._chunks = [Chunk.from_dict(item) for item in self._metadata.get("chunks", [])]
        if len(self._chunks) != int(self._index.ntotal):
            raise ValueError("FAISS index and metadata chunk counts do not match")
        if self.dimension != int(self._index.d):
            raise ValueError("FAISS index and metadata dimensions do not match")
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = int(self._metadata.get("ef_search", 64))

    @classmethod
    def build(
        cls,
        path: Path,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Sequence[float]],
        model_name: str,
        query_instruction: str = "",
        max_length: int = 512,
        pooling: str = "cls",
        hnsw_m: int = 32,
        ef_construction: int = 80,
        ef_search: int = 64,
    ) -> "FaissVectorStore":
        if hnsw_m <= 0 or ef_construction <= 0 or ef_search <= 0:
            raise ValueError("HNSW parameters must be positive")
        path = Path(path)
        metadata_path = Path(f"{path}.meta.json")
        if path.exists() or metadata_path.exists():
            raise FileExistsError(f"FAISS index already exists: {path}")

        chunk_list = list(chunks)
        matrix = _normalise_matrix(embeddings)
        if not chunk_list:
            raise ValueError("cannot build a FAISS index without chunks")
        if len(chunk_list) != matrix.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")

        faiss = _require_faiss()
        index = faiss.IndexHNSWFlat(
            int(matrix.shape[1]),
            int(hnsw_m),
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = int(ef_construction)
        index.hnsw.efSearch = int(ef_search)
        index.add(matrix)

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        metadata = {
            "version": cls.VERSION,
            "backend": "faiss-hnsw",
            "index_type": "IndexHNSWFlat",
            "metric": "inner-product-on-l2-normalized-vectors",
            "model_name": str(model_name),
            "query_instruction": str(query_instruction),
            "max_length": int(max_length),
            "pooling": str(pooling),
            "dimension": int(matrix.shape[1]),
            "count": int(matrix.shape[0]),
            "hnsw_m": int(hnsw_m),
            "ef_construction": int(ef_construction),
            "ef_search": int(ef_search),
            "storage_dtype": "float32",
            "chunks": [chunk.__dict__ for chunk in chunk_list],
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return cls(path)

    @property
    def dimension(self) -> int:
        return int(self._metadata["dimension"])

    @property
    def model_name(self) -> str:
        return str(self._metadata.get("model_name") or "")

    @property
    def query_instruction(self) -> str:
        return str(self._metadata.get("query_instruction") or "")

    @property
    def max_length(self) -> int:
        return int(self._metadata.get("max_length", 512))

    @property
    def pooling(self) -> str:
        return str(self._metadata.get("pooling") or "cls")

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    @property
    def index_type(self) -> str:
        return str(self._metadata.get("index_type") or "")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = _normalise_matrix([query_embedding])
        if query.shape[1] != self.dimension:
            raise ValueError("query embedding dimension does not match the index")
        scores, row_ids = self._index.search(query, min(top_k, self.count))
        scored: list[tuple[float, Chunk]] = []
        for score, row_id in zip(scores[0], row_ids[0]):
            if int(row_id) < 0:
                continue
            if min_score is not None and float(score) < min_score:
                continue
            scored.append((float(score), self._chunks[int(row_id)]))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(rank=rank, score=score, chunk=chunk)
            for rank, (score, chunk) in enumerate(scored, start=1)
        ]
