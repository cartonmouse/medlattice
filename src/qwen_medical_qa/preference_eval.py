"""Model-side scoring helpers for pairwise preference evaluation."""

from __future__ import annotations

from typing import Any


def mean_completion_logprob(logits: Any, input_ids: Any, prompt_length: int) -> float:
    """Return average token log-probability for completion tokens only."""

    import torch

    if logits.ndim != 3 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("expected logits [1, length, vocab] and input_ids [1, length]")
    sequence_length = int(input_ids.shape[1])
    if prompt_length <= 0 or prompt_length >= sequence_length:
        raise ValueError("prompt_length must leave at least one completion token")
    shifted_logits = logits[:, :-1, :].float()
    shifted_targets = input_ids[:, 1:]
    token_log_probs = torch.log_softmax(shifted_logits, dim=-1)
    start = prompt_length - 1
    selected = token_log_probs[:, start:, :].gather(
        dim=-1,
        index=shifted_targets[:, start:].unsqueeze(-1),
    )
    return float(selected.mean().item())
