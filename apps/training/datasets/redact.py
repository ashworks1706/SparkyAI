"""Strip PII from examples before they enter data/. Deterministic, regex-based, no model."""

from __future__ import annotations

import re
from typing import Any

from training.core.types import Message, TrainingExample

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"), "[phone]"),
    (re.compile(r"(?<!\d)\d{17,20}(?!\d)"), "[discord-id]"),
    (re.compile(r"(?<![A-Za-z0-9])\d{10}(?![A-Za-z0-9])"), "[asu-id]"),
    (re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{20,}\b"), "[token]"),
]


def redact_text(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_call(call: dict[str, Any]) -> dict[str, Any]:
    """Rewrites a tool call's values. Keys and ids are structure, not user text."""
    return {k: redact_text(v) if isinstance(v, str) and k != "id" else v for k, v in call.items()}


def redact_message(m: Message) -> Message:
    return m.model_copy(
        update={
            "content": redact_text(m.content),
            "tool_calls": [_redact_call(c) for c in m.tool_calls],
        }
    )


def redact_example(ex: TrainingExample) -> TrainingExample:
    return ex.model_copy(
        update={
            "messages": [redact_message(m) for m in ex.messages],
            "response": redact_message(ex.response),
            "user_id": None,
        }
    )
