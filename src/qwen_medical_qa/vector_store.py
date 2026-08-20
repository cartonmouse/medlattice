"""SQLite-backed persistent vector storage for the RAG retrieval stage."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from .rag import Chunk, RetrievalResult


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("embedding vector cannot have zero norm")
    return [value / norm for value in values]


def _pack_vector(vector: Sequence[float]) -> bytes:
    values = [float(value) for value in vector]
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(payload: bytes, dimension: int) -> tuple[float, ...]:
    expected_size = 4 * dimension
    if len(payload) != expected_size:
        raise ValueError(
            f"stored vector has {len(payload)} bytes; expected {expected_size}"
        )
    return struct.unpack(f"<{dimension}f", payload)


class SqliteVectorStore:
    """A small persistent cosine-search store backed by SQLite.

    SQLite stores chunk metadata and normalized vectors as float32 BLOBs. Search
    is intentionally a transparent scan, which is appropriate for the demo and
    gives the project a stable storage seam before introducing an ANN backend.
    """

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._metadata = self._read_metadata()
        if int(self._metadata.get("version", 0)) != self.VERSION:
            raise ValueError(
                f"unsupported vector store version: {self._metadata.get('version')}"
            )
        self._dimension = int(self._metadata.get("dimension", 0))
        if self._dimension <= 0:
            raise ValueError("vector store dimension must be positive")

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
    ) -> "SqliteVectorStore":
        path = Path(path)
        if path.exists():
            raise FileExistsError(
                f"vector store already exists: {path}; choose a new output path"
            )

        chunk_list = list(chunks)
        vector_list = [_normalize(embedding) for embedding in embeddings]
        if not chunk_list:
            raise ValueError("cannot build a vector store without chunks")
        if len(chunk_list) != len(vector_list):
            raise ValueError("chunks and embeddings must have the same length")
        dimensions = {len(vector) for vector in vector_list}
        if len(dimensions) != 1 or not dimensions or 0 in dimensions:
            raise ValueError("embeddings must have one non-zero dimension")
        dimension = dimensions.pop()

        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": str(cls.VERSION),
            "model_name": str(model_name),
            "query_instruction": str(query_instruction),
            "max_length": str(int(max_length)),
            "pooling": str(pooling),
            "dimension": str(dimension),
            "metric": "inner-product-on-l2-normalized-vectors",
            "storage_dtype": "float32-le",
        }
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = DELETE;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE chunks (
                    row_id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    license TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_blob BLOB NOT NULL
                );
                CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, doc_id, title, text, source, license,
                    start_offset, end_offset, metadata_json, vector_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.text,
                        chunk.source,
                        chunk.license,
                        chunk.start,
                        chunk.end,
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        _pack_vector(vector),
                    )
                    for chunk, vector in zip(chunk_list, vector_list)
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return cls(path)

    def _read_metadata(self) -> dict[str, str]:
        connection = sqlite3.connect(self.path)
        try:
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"invalid vector store: {self.path}") from exc
        finally:
            connection.close()
        return {str(key): str(value) for key, value in rows}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._metadata.get("model_name", "")

    @property
    def query_instruction(self) -> str:
        return self._metadata.get("query_instruction", "")

    @property
    def max_length(self) -> int:
        return int(self._metadata.get("max_length", 512))

    @property
    def pooling(self) -> str:
        return self._metadata.get("pooling", "cls")

    @property
    def count(self) -> int:
        connection = sqlite3.connect(self.path)
        try:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        finally:
            connection.close()

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = _normalize(query_embedding)
        if len(query) != self.dimension:
            raise ValueError("query embedding dimension does not match the store")

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT chunk_id, doc_id, title, text, source, license,
                       start_offset, end_offset, metadata_json, vector_blob
                FROM chunks
                """
            ).fetchall()
        finally:
            connection.close()

        scored: list[tuple[float, Chunk]] = []
        for row in rows:
            vector = _unpack_vector(row["vector_blob"], self.dimension)
            score = sum(left * right for left, right in zip(query, vector))
            if min_score is not None and score < min_score:
                continue
            chunk = Chunk(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                text=str(row["text"]),
                source=str(row["source"]),
                license=str(row["license"]),
                start=int(row["start_offset"]),
                end=int(row["end_offset"]),
                metadata=json.loads(str(row["metadata_json"])),
            )
            scored.append((float(score), chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(rank=index, score=score, chunk=chunk)
            for index, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]
