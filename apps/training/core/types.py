"""Types for the engine evals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Expectation(BaseModel):
    """What a golden case expects of the engine. Every field is optional."""

    status: str | None = None
    source_key: str | None = None
    tool: str | None = None
    tool_args_contain: dict[str, str] = Field(default_factory=dict)
    policy: Literal["allow", "deny", "confirm"] | None = None
    refuse: bool = False
    clarify: bool = False
    mentions: list[str] = Field(default_factory=list)
    not_mentions: list[str] = Field(default_factory=list)
    remembers: str | None = None
    max_latency_ms: int | None = None


class EvalCase(BaseModel):
    """A hand-written question with expectations. follow_up makes a two-turn case."""

    id: str
    suites: list[str]
    question: str
    roles: list[str] = Field(default_factory=list)
    follow_up: str | None = None
    expect: Expectation = Field(default_factory=Expectation)


class Citation(BaseModel):
    """One source under an answer. Mirrors the engine wire shape."""

    key: str = ""
    title: str
    url: str | None = None


class TurnResult(BaseModel):
    """The engine answer to one turn plus its trace."""

    request_id: str
    conversation_id: str
    status: str
    text: str
    citations: list[Citation]
    steps: int
    tokens: int
    latency_ms: int
    events: list[dict[str, Any]]


class Score(BaseModel):
    passed: bool
    detail: str


class CaseResult(BaseModel):
    case_id: str
    suite: str
    score: Score
    request_id: str


class SuiteReport(BaseModel):
    suite: str
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class EvalReport(BaseModel):
    engine_url: str
    cases: int
    results: list[CaseResult]
    suites: list[SuiteReport]


class RunnerError(RuntimeError):
    """A case could not be run: no cases, engine unreachable, or no trace for the request."""
