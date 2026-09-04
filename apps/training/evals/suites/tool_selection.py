"""Did the engine call the tool the case expects (and nothing riskier)?"""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    called = [e["tool"] for t in turns for e in t.events if e["kind"] == "tool_call"]
    expected = case.expect.tool
    if expected is None:
        return Score(passed=not called, detail=f"expected no tool, called {called}")
    return Score(passed=expected in called, detail=f"expected {expected}, called {called}")
