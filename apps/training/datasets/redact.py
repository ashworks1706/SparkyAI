"""Strip PII from examples before they enter data/. Deterministic, regex-based, no model."""

from __future__ import annotations

import re

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


def redact_message(m: Message) -> Message:
    return m.model_copy(update={"content": redact_text(m.content)})


def redact_example(ex: TrainingExample) -> TrainingExample:
    return ex.model_copy(
        update={
            "messages": [redact_message(m) for m in ex.messages],
            "response": redact_message(ex.response),
            "user_id": None,
        }
    )
