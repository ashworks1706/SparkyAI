"""Did the expected tool receive arguments containing the expected values?"""

from __future__ import annotations

import json

from training.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score | None:
    want = case.expect.tool_args_contain
    if not want:
        return None
    calls = [
        e
        for t in turns
        for e in t.events
        if e["kind"] == "tool_call" and e["tool"] == case.expect.tool
    ]
    if not calls:
        return Score(passed=False, detail=f"{case.expect.tool} was not called")
    for call in calls:
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        if all(k in args and v.lower() in json.dumps(args[k]).lower() for k, v in want.items()):
            return Score(passed=True, detail=f"arguments contained {want}")
    return Score(passed=False, detail=f"no call contained {want}")
