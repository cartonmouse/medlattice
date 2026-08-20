"""Conservative, transparent safety gates for the demo RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


ABSTENTION_ANSWER = "资料不足，无法判断。"

# This is deliberately a small baseline rule set, not a medical safety model.
HIGH_RISK_PATTERNS = (
    "我的症状",
    "我的病",
    "帮我诊断",
    "给我诊断",
    "怎么治疗",
    "如何治疗",
    "吃什么药",
    "服用哪种药",
    "药物剂量",
    "用药剂量",
    "是否需要手术",
    "生存时间",
    "能活多久",
)


@dataclass(frozen=True)
class SafetyDecision:
    abstain: bool
    reason: str
    matched_pattern: str | None = None
    top_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstain": self.abstain,
            "reason": self.reason,
            "matched_pattern": self.matched_pattern,
            "top_score": self.top_score,
        }


def _item_score(item: Any) -> float | None:
    if isinstance(item, dict):
        value = item.get("score")
    else:
        value = getattr(item, "score", None)
    return float(value) if value is not None else None


def assess_question(
    question: str,
    results: Iterable[Any],
    *,
    min_score: float | None = None,
) -> SafetyDecision:
    """Return a conservative abstention decision for a question and context.

    The policy is intentionally explainable: high-risk patient-specific/action
    requests abstain before generation; missing or low-confidence retrieval also
    abstains. It is a demo gate and must not be treated as a clinical safeguard.
    """
    normalized = "".join(str(question).split())
    for pattern in HIGH_RISK_PATTERNS:
        if pattern in normalized:
            return SafetyDecision(
                abstain=True,
                reason="high-risk-intent",
                matched_pattern=pattern,
            )

    materialized = list(results)
    top_score = _item_score(materialized[0]) if materialized else None
    if not materialized:
        return SafetyDecision(
            abstain=True,
            reason="no-retrieved-context",
            top_score=top_score,
        )
    if min_score is not None and (top_score is None or top_score < min_score):
        return SafetyDecision(
            abstain=True,
            reason="below-retrieval-threshold",
            top_score=top_score,
        )
    return SafetyDecision(
        abstain=False,
        reason="within-demo-scope",
        top_score=top_score,
    )
