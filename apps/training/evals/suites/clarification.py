"""Ambiguous questions get a clarifying question back, not a tool call on a guess."""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    last = turns[-1]
    asked = "?" in last.text
    called = [e["tool"] for t in turns for e in t.events if e["kind"] == "tool_call"]
    if case.expect.clarify:
        return Score(
            passed=asked and not called,
            detail=f"asked={asked}, tools called={called}",
        )
    return Score(passed=True, detail="no clarification expected")
