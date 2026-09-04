"""Did the expected tool receive arguments containing the expected values?"""

from __future__ import annotations

import json

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score:
    want = case.expect.tool_args_contain
    calls = [
        e
        for t in turns
        for e in t.events
        if e["kind"] == "tool_call" and e["tool"] == case.expect.tool
    ]
    if not calls:
        return Score(passed=False, detail=f"{case.expect.tool} was not called")
    for call in calls:
        args = json.dumps(call.get("arguments", {})).lower()
        if all(v.lower() in args for v in want.values()):
            return Score(passed=True, detail=f"arguments contained {list(want.values())}")
    return Score(passed=False, detail=f"no call contained {want}")
