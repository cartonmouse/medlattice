"""Optional Transformer Cross-Encoder reranking for retrieved candidates."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .rag import RetrievalResult
from .reranker import RerankedResult


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
ScoreFunction = Callable[[str, Sequence[str]], Sequence[float]]


def format_passage(candidate: RetrievalResult) -> str:
    """Render one candidate in the stable query-passage format for the model."""
    title = candidate.chunk.title.strip()
    text = candidate.chunk.text.strip()
    return f"{title}\n{text}" if title else text


def _build_results(
    candidates: Sequence[RetrievalResult],
    scores: Sequence[float],
    top_k: int | None,
) -> list[RerankedResult]:
    if len(candidates) != len(scores):
        raise ValueError("reranker returned a score for a different number of candidates")

    ranked = [
        (float(score), candidate.score, candidate.chunk)
        for candidate, score in zip(candidates, scores)
    ]
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2].chunk_id))
    limit = top_k if top_k is not None else len(ranked)
    return [
        RerankedResult(
            rank=index,
            score=score,
            chunk=chunk,
            retrieval_score=retrieval_score,
        )
        for index, (score, retrieval_score, chunk) in enumerate(ranked[:limit], start=1)
    ]


class TransformerCrossEncoderReranker:
    """Score query-passage pairs with a Transformer sequence classifier.

    The model score is a raw classification logit. It is suitable for ranking
    within one candidate set, but it should not be interpreted as a probability
    or compared across different reranker models. ``score_fn`` is an internal
    seam used by tests and lightweight experiments; production use loads the
    configured Transformers model lazily at construction time.
    """

    DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAME
    score_type = "logit"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        local_files_only: bool = False,
        score_fn: ScoreFunction | None = None,
    ) -> None:
        if not str(model_name).strip():
            raise ValueError("model_name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.local_files_only = bool(local_files_only)
        self.device = str(device)
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._score_fn = score_fn
        if self._score_fn is None:
            self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = self._resolve_device(torch, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
        self._score_fn = self._score_with_model

    @staticmethod
    def _resolve_device(torch: object, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def _score_with_model(self, query: str, passages: Sequence[str]) -> list[float]:
        if self._torch is None or self._tokenizer is None or self._model is None:
            raise RuntimeError("Transformer reranker model is not loaded")

        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = list(passages[start : start + self.batch_size])
                encoded = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device) for key, value in encoded.items()
                }
                output = self._model(**encoded)
                logits = output.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                elif logits.ndim != 1:
                    raise ValueError(
                        "cross-encoder must return one logit per query-passage pair"
                    )
                scores.extend(logits.detach().float().cpu().tolist())
        return scores

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
        if self._score_fn is None:
            raise RuntimeError("reranker score function is not initialized")

        passages = [format_passage(candidate) for candidate in items]
        scores = self._score_fn(str(query).strip(), passages)
        return _build_results(items, scores, top_k)
