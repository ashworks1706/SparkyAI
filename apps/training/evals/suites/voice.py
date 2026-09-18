"""Scores whether the answer stays in the student's language instead of naming its own machinery.

A student cannot call a tool and does not know what one is called. An answer that names
search_live, a source key or a parameter has handed the work back in the one way the loop
cannot recover from.
"""

from __future__ import annotations

from training.core.types import EvalCase, Score, TurnResult

#: Named in every case, on top of whatever the case itself forbids.
INTERNALS = ("search_live", "search_knowledge", "run_sandbox", "tool call", "source key")


def score(case: EvalCase, turns: list[TurnResult]) -> Score | None:
    forbidden = list(INTERNALS) + list(case.expect.not_mentions)
    last = turns[-1]
    text = last.text.lower()
    named = [word for word in forbidden if word.lower() in text]
    if named:
        return Score(passed=False, detail=f"answer names {named}")
    return Score(passed=True, detail="answer names nothing internal")
