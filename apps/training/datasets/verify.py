"""Keep only examples worth training on: well-formed, non-empty, not duplicates."""

from __future__ import annotations

import hashlib
import json

from training.core.types import TrainingExample


def fingerprint(ex: TrainingExample) -> str:
    payload = json.dumps(
        {"m": [m.model_dump() for m in ex.messages], "r": ex.response.model_dump()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def reject_reason(ex: TrainingExample) -> str | None:
    if not ex.messages:
        return "no messages"
    if ex.messages[0].role != "system":
        return "first message is not system"
    if not any(m.role == "user" for m in ex.messages):
        return "no user turn"
    if not ex.response.content.strip() and not ex.response.tool_calls:
        return "empty response"
    for call in ex.response.tool_calls:
        if not isinstance(call.get("name"), str) or not call.get("name"):
            return "tool call without a name"
    return None


def verify(examples: list[TrainingExample]) -> tuple[list[TrainingExample], dict[str, int]]:
    """Returns (kept, reasons) where reasons counts every rejection cause."""
    kept: list[TrainingExample] = []
    reasons: dict[str, int] = {}
    seen: set[str] = set()
    for ex in examples:
        reason = reject_reason(ex)
        if reason is None:
            fp = fingerprint(ex)
            if fp in seen:
                reason = "duplicate"
            else:
                seen.add(fp)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept.append(ex)
    return kept, reasons
