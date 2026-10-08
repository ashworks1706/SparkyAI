"""Ambiguous questions get a clarifying question back and no tool call."""

from __future__ import annotations

from evals.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score | None:
    """Passes when the last turn asks a question and no tool ran; None unless clarify is set."""
    last = turns[-1]
    asked = "?" in last.text
    called = [c["tool"] for t in turns for c in t.tool_calls()]
    if case.expect.clarify:
        return Score(
            passed=asked and not called,
            detail=f"asked={asked}, tools called={called}",
        )
    return None
