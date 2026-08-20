"""Transformer embedding retrieval for the RAG comparison stage."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .rag import Chunk, RetrievalResult


DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0:
        raise ValueError("embedding vector cannot have zero norm")
    return [float(value) / norm for value in vector]


class DenseRetriever:
    """A serializable dense-vector retriever using cosine similarity."""

    VERSION = 1

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Sequence[float]],
        model_name: str,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        max_length: int = 512,
        pooling: str = "cls",
    ) -> None:
        self.chunks = list(chunks)
        raw_embeddings = [list(map(float, embedding)) for embedding in embeddings]
        if not self.chunks:
            raise ValueError("cannot create a dense retriever without chunks")
        if len(self.chunks) != len(raw_embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        dimensions = {len(embedding) for embedding in raw_embeddings}
        if len(dimensions) != 1 or not dimensions or 0 in dimensions:
            raise ValueError("embeddings must have one non-zero dimension")
        self.embeddings = [_normalize(embedding) for embedding in raw_embeddings]
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.pooling = pooling

    @property
    def dimension(self) -> int:
        return len(self.embeddings[0])

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
            raise ValueError("query embedding dimension does not match the index")

        scored = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = sum(left * right for left, right in zip(query, embedding))
            if min_score is None or score >= min_score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(rank=index, score=score, chunk=chunk)
            for index, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "retriever": "dense-cosine",
            "model_name": self.model_name,
            "query_instruction": self.query_instruction,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "dimension": self.dimension,
            "chunks": [chunk.__dict__ for chunk in self.chunks],
            "embeddings": self.embeddings,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseRetriever":
        if int(payload.get("version", 0)) != cls.VERSION:
            raise ValueError(f"unsupported dense index version: {payload.get('version')}")
        chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        return cls(
            chunks=chunks,
            embeddings=payload.get("embeddings", []),
            model_name=str(payload.get("model_name") or DEFAULT_MODEL_NAME),
            query_instruction=str(payload.get("query_instruction") or ""),
            max_length=int(payload.get("max_length", 512)),
            pooling=str(payload.get("pooling") or "cls"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "DenseRetriever":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class TransformerTextEncoder:
    """Small wrapper around AutoTokenizer/AutoModel with batched CLS pooling."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        local_files_only: bool = False,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def encode(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        prepared = [str(text) for text in texts]
        if is_query and self.query_instruction:
            prepared = [self.query_instruction + text for text in prepared]

        embeddings: list[list[float]] = []
        with self.torch.inference_mode():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                pooled = output.last_hidden_state[:, 0, :]
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.extend(pooled.detach().cpu().float().tolist())
        return embeddings
