"""Scores whether the expected tool received arguments containing the expected values."""

from __future__ import annotations

import json

from evals.core.types import EvalCase, Score, TurnResult


def score(case: EvalCase, turns: list[TurnResult]) -> Score | None:
    """Passes when a call of the expected tool carries the expected values; None without them."""
    want = case.expect.tool_args_contain
    if not want:
        return None
    calls = [c for t in turns for c in t.tool_calls() if c["tool"] == case.expect.tool]
    if not calls:
        return Score(passed=False, detail=f"{case.expect.tool} was not called")
    unreadable = 0
    for call in calls:
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                unreadable += 1
                continue
        if not isinstance(args, dict):
            unreadable += 1
            continue
        if all(k in args and v.lower() in json.dumps(args[k]).lower() for k, v in want.items()):
            return Score(passed=True, detail=f"arguments contained {want}")
    detail = f"no call contained {want}"
    if unreadable:
        detail += f"; {unreadable} of {len(calls)} calls had arguments that are not a JSON object"
    return Score(passed=False, detail=detail)
