"""Evaluation helpers for citation-aware RAG answers."""

from __future__ import annotations

import re
from typing import Any


_CITATION_PATTERN = re.compile(r"\[([^\]]+#chunk-\d+)\]")


def extract_citation_ids(answer: str) -> list[str]:
    """Extract chunk citation identifiers such as ``[term-symptom#chunk-000]``."""
    return _CITATION_PATTERN.findall(answer or "")


def _compact(text: str) -> str:
    return "".join(str(text).split()).casefold()


def score_rag_answers(rows: list[dict[str, Any]], k: int = 3) -> dict[str, Any]:
    """Score retrieval hit, answer substring, and citation presence separately."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not rows:
        raise ValueError("cannot score an empty RAG answer list")

    details = []
    retrieval_hits = 0
    answer_hits = 0
    citation_hits = 0
    grounded_hits = 0
    for row in rows:
        relevant = {str(value) for value in row.get("relevant_doc_ids", [])}
        retrieved = row.get("retrieved") or []
        retrieved_doc_ids = [str(item.get("doc_id") or "") for item in retrieved]
        retrieved_chunk_ids = {str(item.get("chunk_id") or "") for item in retrieved}
        first_relevant_rank = next(
            (
                index
                for index, doc_id in enumerate(retrieved_doc_ids, start=1)
                if doc_id in relevant
            ),
            None,
        )
        retrieval_hit = first_relevant_rank is not None and first_relevant_rank <= k
        answer = str(row.get("answer") or "")
        expected = str(row.get("expected_answer") or "")
        answer_hit = bool(expected) and _compact(expected) in _compact(answer)
        citations = extract_citation_ids(answer)
        valid_citation = any(citation in retrieved_chunk_ids for citation in citations)
        if retrieval_hit:
            retrieval_hits += 1
        if answer_hit:
            answer_hits += 1
        if valid_citation:
            citation_hits += 1
        if answer_hit and valid_citation:
            grounded_hits += 1
        details.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "expected_answer": expected,
                "answer": answer,
                "retrieved_doc_ids": retrieved_doc_ids,
                "first_relevant_rank": first_relevant_rank,
                "retrieval_hit_at_k": retrieval_hit,
                "citation_ids": citations,
                "valid_citation": valid_citation,
                "answer_contains_expected": answer_hit,
                "grounded_answer": answer_hit and valid_citation,
            }
        )

    total = len(rows)
    return {
        "task": "rag_answer_grounding_demo",
        "total": total,
        "k": k,
        "retrieval_hit_rate_at_k": round(retrieval_hits / total, 4),
        "answer_contains_expected_rate": round(answer_hits / total, 4),
        "valid_citation_rate": round(citation_hits / total, 4),
        "grounded_answer_rate": round(grounded_hits / total, 4),
        "details": details,
    }
