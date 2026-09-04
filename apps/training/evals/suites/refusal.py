"""Out-of-scope or unanswerable questions must be declined, not guessed."""

from __future__ import annotations

import re

from training.core.types import EvalCase, Score, TurnResult

_DECLINE = re.compile(
    r"(can(?:no|')t|cannot|unable|not (?:able|available|find|have)|"
    r"no (?:evidence|information|source)|don't have|do not have|outside|not something i)",
    re.IGNORECASE,
)


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    last = turns[-1]
    declined = bool(_DECLINE.search(last.text))
    guessed = bool(last.citations) and not declined
    if case.expect.refuse:
        return Score(
            passed=declined and not guessed,
            detail="declined" if declined else f"did not decline: {last.text[:80]!r}",
        )
    return Score(passed=not declined, detail="answered" if not declined else "declined wrongly")
