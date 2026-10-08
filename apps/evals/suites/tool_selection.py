"""Scores whether the engine called the tool the case expects."""

from __future__ import annotations

from evals.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score | None:
    """Passes when the expected tool ran, or when none ran and none was expected."""
    called = [c["tool"] for t in turns for c in t.tool_calls()]
    expected = case.expect.tool
    if expected is None:
        return Score(passed=not called, detail=f"expected no tool, called {called}")
    return Score(passed=expected in called, detail=f"expected {expected}, called {called}")
