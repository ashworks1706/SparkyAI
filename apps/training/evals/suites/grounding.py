"""Is the answer cited from the expected source, and does it mention what it should?"""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    last = turns[-1]
    problems: list[str] = []
    if last.status != "answered":
        problems.append(f"status {last.status}")
    if case.expect.source_key and not any(
        c.startswith(case.expect.source_key) for c in last.citations
    ):
        problems.append(f"no citation from {case.expect.source_key}; got {last.citations[:2]}")
    text = last.text.lower()
    missing = [m for m in case.expect.mentions if m.lower() not in text]
    if missing:
        problems.append(f"answer does not mention {missing}")
    return Score(passed=not problems, detail="; ".join(problems) or "cited and on point")
