"""Transparent candidate rerankers for the RAG retrieval pipeline."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .rag import Chunk, RetrievalResult, tokenize


@dataclass
class RerankedResult:
    """A reranked result retaining the original dense score for analysis."""

    rank: int
    score: float
    chunk: Chunk
    retrieval_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "retrieval_score": round(self.retrieval_score, 6),
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "license": self.chunk.license,
        }


def describe_reranker(reranker: object | None) -> dict[str, object | None]:
    """Return stable metadata without requiring a concrete reranker type."""
    return {
        "reranker_model": getattr(reranker, "model_name", None),
        "reranker_device": getattr(reranker, "device", None),
        "reranker_batch_size": getattr(reranker, "batch_size", None),
        "reranker_max_length": getattr(reranker, "max_length", None),
        "reranker_score_type": getattr(reranker, "score_type", None),
    }


class BM25Reranker:
    """A dependency-free BM25 reranker over a dense-retrieved candidate set."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = float(k1)
        self.b = float(b)

    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        items = list(candidates)
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if not items:
            return []

        query_terms = tokenize(query)
        documents = [
            tokenize(f"{item.chunk.title} {item.chunk.text}") for item in items
        ]
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        average_length = sum(len(document) for document in documents) / len(documents)
        average_length = average_length or 1.0
        total_documents = len(documents)

        scored: list[tuple[float, float, Chunk]] = []
        for item, document in zip(items, documents):
            frequencies = Counter(document)
            length = len(document) or 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_count = document_frequency[term]
                idf = math.log(
                    1.0
                    + (total_documents - document_count + 0.5)
                    / (document_count + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / average_length
                )
                score += idf * frequency * (self.k1 + 1.0) / denominator
            scored.append((score, item.score, item.chunk))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2].chunk_id))
        limit = top_k if top_k is not None else len(scored)
        return [
            RerankedResult(
                rank=index,
                score=score,
                chunk=chunk,
                retrieval_score=retrieval_score,
            )
            for index, (score, retrieval_score, chunk) in enumerate(scored[:limit], start=1)
        ]
