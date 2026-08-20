"""Shared prompt construction for baseline inference and SFT data."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = "你是一个用于研究评测的中文医疗问答模型。回答仅供学习演示，不构成医疗建议。"
CHOICE_INSTRUCTION = "这是一个教育性医学术语选择题。请只输出一个选项字母，不要输出解释。"


def format_user_prompt(record: dict[str, Any]) -> str:
    question = str(record["question"]).strip()
    choices = record.get("choices")
    if not choices:
        return question
    if not isinstance(choices, dict):
        raise ValueError(f"choices must be an object for record {record.get('id', '<unknown>')}")

    options = "\n".join(f"{key}. {value}" for key, value in choices.items())
    return f"{question}\n\n选项：\n{options}\n\n{CHOICE_INSTRUCTION}"


def build_messages(record: dict[str, Any], assistant_content: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_prompt(record)},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages
