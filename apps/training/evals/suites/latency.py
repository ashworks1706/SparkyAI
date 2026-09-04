"""Wall-clock per turn stays under the case's budget."""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    limit = case.expect.max_latency_ms
    if limit is None:
        return Score(passed=True, detail="no latency budget")
    worst = max(t.latency_ms for t in turns)
    return Score(passed=worst <= limit, detail=f"{worst} ms vs {limit} ms budget")
