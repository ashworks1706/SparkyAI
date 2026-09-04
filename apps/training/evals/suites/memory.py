"""A fact stated in the first turn is used in the follow-up."""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    if len(turns) < 2:
        return Score(passed=False, detail="memory cases need a follow_up")
    want = (case.expect.remembers or "").lower()
    if not want:
        return Score(passed=False, detail="case has no `remembers` expectation")
    hit = want in turns[-1].text.lower()
    return Score(passed=hit, detail=f"{'found' if hit else 'missing'} {want!r} in follow-up")
