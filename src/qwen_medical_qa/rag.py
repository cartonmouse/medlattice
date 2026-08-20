"""Transparent, dependency-free retrieval primitives for the RAG v0 stage."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


@dataclass
class Document:
    """A source document that can be split into retrieval chunks."""

    doc_id: str
    title: str
    text: str
    source: str = ""
    license: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Document":
        doc_id = str(payload.get("doc_id") or payload.get("id") or "").strip()
        title = str(payload.get("title") or doc_id).strip()
        text = str(payload.get("text") or "").strip()
        if not doc_id:
            raise ValueError("document is missing doc_id")
        if not text:
            raise ValueError(f"document {doc_id!r} has empty text")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"document {doc_id!r} metadata must be an object")
        return cls(
            doc_id=doc_id,
            title=title,
            text=text,
            source=str(payload.get("source") or "").strip(),
            license=str(payload.get("license") or "").strip(),
            metadata=metadata,
        )


@dataclass
class Chunk:
    """A retrieval unit retaining its source document and character offsets."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str = ""
    license: str = ""
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            title=str(payload.get("title") or ""),
            text=str(payload["text"]),
            source=str(payload.get("source") or ""),
            license=str(payload.get("license") or ""),
            start=int(payload.get("start", 0)),
            end=int(payload.get("end", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class RetrievalResult:
    """A ranked retrieval result with a cosine similarity score."""

    rank: int
    score: float
    chunk: Chunk

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "license": self.chunk.license,
        }


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese characters deterministically."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def chunk_document(
    document: Document,
    chunk_size: int = 160,
    chunk_overlap: int = 32,
) -> list[Chunk]:
    """Split a document by characters while retaining a small overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    text = re.sub(r"\s+", " ", document.text).strip()
    if not text:
        raise ValueError(f"document {document.doc_id!r} has empty text")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#chunk-{index:03d}",
                doc_id=document.doc_id,
                title=document.title,
                text=text[start:end],
                source=document.source,
                license=document.license,
                start=start,
                end=end,
                metadata=document.metadata.copy(),
            )
        )
        if end == len(text):
            break
        start = end - chunk_overlap
        index += 1
    return chunks


class TfidfRetriever:
    """A small sparse TF-IDF cosine retriever for the reproducible RAG baseline."""

    VERSION = 1

    def __init__(self, chunks: Iterable[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []
        if chunks is not None:
            self.fit(chunks)

    def fit(self, chunks: Iterable[Chunk]) -> "TfidfRetriever":
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("cannot fit a retriever without chunks")

        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            document_frequency.update(set(tokenize(chunk.text)))

        count = len(self.chunks)
        self._idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._build_vectors()
        return self

    def _build_vectors(self) -> None:
        self._vectors = []
        self._norms = []
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            token_count = len(tokens) or 1
            vector = {
                term: (frequency / token_count) * self._idf[term]
                for term, frequency in counts.items()
                if term in self._idf
            }
            self._vectors.append(vector)
            self._norms.append(math.sqrt(sum(value * value for value in vector.values())))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return positive-scoring chunks ordered deterministically."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.chunks:
            return []

        tokens = tokenize(query)
        counts = Counter(tokens)
        token_count = len(tokens) or 1
        query_vector = {
            term: (frequency / token_count) * self._idf[term]
            for term, frequency in counts.items()
            if term in self._idf
        }
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0:
            return []

        scored: list[tuple[float, Chunk]] = []
        for chunk, vector, norm in zip(self.chunks, self._vectors, self._norms):
            if norm == 0:
                continue
            dot = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            if dot <= 0:
                continue
            scored.append((dot / (query_norm * norm), chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(rank=index, score=score, chunk=chunk)
            for index, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "tokenizer": "english-word-plus-chinese-character",
            "idf": self._idf,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TfidfRetriever":
        if int(payload.get("version", 0)) != cls.VERSION:
            raise ValueError(f"unsupported RAG index version: {payload.get('version')}")
        retriever = cls()
        retriever.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        retriever._idf = {str(term): float(value) for term, value in payload.get("idf", {}).items()}
        if not retriever.chunks or not retriever._idf:
            raise ValueError("RAG index must contain chunks and idf values")
        retriever._build_vectors()
        return retriever

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def format_context(results: Iterable[RetrievalResult]) -> str:
    """Format retrieved chunks with stable citation identifiers."""
    rendered = []
    for result in results:
        source = result.chunk.source or "unknown"
        rendered.append(
            f"[{result.chunk.chunk_id}] {result.chunk.title}\n"
            f"{result.chunk.text}\n"
            f"来源：{source}"
        )
    return "\n\n".join(rendered) if rendered else "[无相关资料]"


def build_rag_prompt(question: str, results: Iterable[RetrievalResult]) -> str:
    """Build a citation-aware prompt for a later generation stage."""
    context = format_context(results)
    return (
        "你是一个需要依据资料回答问题的助手。\n"
        "只使用资料中明确出现的信息；资料不足时回答‘资料不足，无法判断’。\n"
        "回答后列出使用的引用编号，不要编造资料中没有的事实。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question.strip()}\n"
        "回答："
    )


def read_documents_jsonl(path: Path) -> list[Document]:
    """Read and validate a JSONL knowledge-base file."""
    documents: list[Document] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = Document.from_dict(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid knowledge-base row at line {line_number}") from exc
        if document.doc_id in seen:
            raise ValueError(f"duplicate document id: {document.doc_id}")
        seen.add(document.doc_id)
        documents.append(document)
    if not documents:
        raise ValueError(f"no documents found in {path}")
    return documents
