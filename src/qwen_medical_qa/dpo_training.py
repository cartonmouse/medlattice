"""Helpers for rendering local preference pairs before passing them to TRL."""

from __future__ import annotations

from typing import Any, Iterable

from qwen_medical_qa.dpo_data import validate_preference_record


def _assistant_content(value: Any, row_id: str, field: str) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{row_id} {field} must contain one message")
    message = value[0]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError(f"{row_id} {field} must contain an assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{row_id} {field} must contain text")
    return content


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template must return non-empty text")
    return rendered


def render_preference_rows(
    rows: Iterable[dict[str, Any]], tokenizer: Any
) -> list[dict[str, str]]:
    """Render conversational records into TRL's standard prompt/chosen/rejected form."""

    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        validate_preference_record(row)
        row_id = str(row["id"])
        prompt = row["prompt"]
        prompt_text = _apply_chat_template(tokenizer, prompt)
        rendered_rows.append(
            {
                "prompt": prompt_text,
                "chosen": _assistant_content(row["chosen"], row_id, "chosen"),
                "rejected": _assistant_content(row["rejected"], row_id, "rejected"),
            }
        )
    if not rendered_rows:
        raise ValueError("no preference rows were rendered")
    return rendered_rows
