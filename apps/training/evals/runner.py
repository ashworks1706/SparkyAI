"""Runs golden cases against the engine's /chat and collects each turn's answer and trace."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from training.core.settings import settings
from training.core.types import EvalCase, TurnResult


class RunnerError(RuntimeError):
    pass


def load_cases(cases_dir: Path | None = None) -> list[EvalCase]:
    cases_dir = cases_dir or settings().training.cases_dir
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                cases.append(EvalCase.model_validate_json(line))
    if not cases:
        raise RunnerError(f"no cases under {cases_dir}")
    return cases


def read_trace(request_id: str, traces_dir: Path | None = None) -> list[dict]:
    path = (traces_dir or settings().training.traces_dir) / f"{request_id}.jsonl"
    if not path.exists():
        raise RunnerError(
            f"trace {path} not found; run the engine from the repo root so traces/ is shared"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ask(
    question: str,
    *,
    roles: list[str],
    conversation_id: str | None,
    user_id: str,
    engine_url: str | None = None,
) -> TurnResult:
    cfg = settings().training
    body = {"user_id": user_id, "roles": roles, "message": question, "channel_id": "eval"}
    if conversation_id:
        body["conversation_id"] = conversation_id
    url = (engine_url or cfg.engine_url).rstrip("/") + "/chat"
    started = time.monotonic()
    try:
        r = httpx.post(url, json=body, timeout=cfg.request_timeout_secs)
    except httpx.HTTPError as e:
        raise RunnerError(f"engine at {url} unreachable: {e}") from e
    latency_ms = int((time.monotonic() - started) * 1000)
    if r.status_code != 200:
        raise RunnerError(f"engine returned {r.status_code}: {r.text[:200]}")
    d = r.json()
    return TurnResult(
        request_id=d["request_id"],
        conversation_id=d["conversation_id"],
        status=d["status"],
        text=d["text"],
        citations=d.get("citations") or [],
        steps=d["steps"],
        tokens=d["tokens"],
        latency_ms=latency_ms,
        events=read_trace(d["request_id"]),
    )


def run_case(case: EvalCase, engine_url: str | None = None) -> list[TurnResult]:
    """One or two turns; the follow-up continues the same conversation."""
    user_id = f"eval-{case.id}"
    first = ask(
        case.question,
        roles=case.roles,
        conversation_id=None,
        user_id=user_id,
        engine_url=engine_url,
    )
    turns = [first]
    if case.follow_up:
        turns.append(
            ask(
                case.follow_up,
                roles=case.roles,
                conversation_id=first.conversation_id,
                user_id=user_id,
                engine_url=engine_url,
            )
        )
    return turns
