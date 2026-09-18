"""Scores whether the answer cites the expected source and mentions the expected terms."""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score | None:
    last = turns[-1]
    problems: list[str] = []
    if last.status != "answered":
        problems.append(f"status {last.status}")
    # A citation is labelled with the page title, so the source it came from is its key.
    # An engine that predates the key field carries an empty one; fall back to the title.
    if case.expect.source_key and not any(
        (c.key or c.title).startswith(case.expect.source_key) for c in last.citations
    ):
        cited = [c.key or c.title for c in last.citations[:2]]
        problems.append(f"no citation from {case.expect.source_key}; got {cited}")
    text = last.text.lower()
    missing = [m for m in case.expect.mentions if m.lower() not in text]
    if missing:
        problems.append(f"answer does not mention {missing}")
    return Score(passed=not problems, detail="; ".join(problems) or "cited and on point")
