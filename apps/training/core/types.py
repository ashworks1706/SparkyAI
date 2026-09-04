"""Types shared across datasets, evals, and post-training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One chat turn, matching the engine's `core::types::message::Message`."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


class TrainingExample(BaseModel):
    """One model call as it happened: the full prompt and the reply. One `llm` span."""

    id: str
    messages: list[Message]
    response: Message
    model: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    tool_count: int = 0


class Expectation(BaseModel):
    """What a golden case expects of the engine. Every field is optional; suites read the
    ones they care about."""

    status: str | None = None
    source_key: str | None = None
    tool: str | None = None
    tool_args_contain: dict[str, str] = Field(default_factory=dict)
    policy: Literal["allow", "deny", "confirm"] | None = None
    refuse: bool = False
    clarify: bool = False
    mentions: list[str] = Field(default_factory=list)
    remembers: str | None = None
    max_latency_ms: int | None = None


class EvalCase(BaseModel):
    """A hand-written question with expectations. `follow_up` makes a two-turn case."""

    id: str
    suites: list[str]
    question: str
    roles: list[str] = Field(default_factory=list)
    follow_up: str | None = None
    expect: Expectation = Field(default_factory=Expectation)


class TurnResult(BaseModel):
    """The engine's answer to one turn plus its trace."""

    request_id: str
    conversation_id: str
    status: str
    text: str
    citations: list[str]
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


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int
    per_device_batch_size: int
    gradient_accumulation: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    seed: int


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gguf_quant: str


class SftConfig(BaseModel):
    """`configs/train/sft.yaml`. Every knob is explicit; an unknown or missing key fails."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    dataset: Path
    output_dir: Path
    max_seq_length: int
    load_in_4bit: bool
    lora: LoraConfig
    train: TrainConfig
    export: ExportConfig


class SftPlan(BaseModel):
    """What `train sft` would do, printed before it does it."""

    base_model: str
    dataset: Path
    output_dir: Path
    examples: int
    with_tool_calls: int
    max_seq_length: int
    epochs: int
    gguf_quant: str


class ExportError(RuntimeError):
    """Phoenix answered with something other than complete `llm` spans."""


class RunnerError(RuntimeError):
    """A case could not be run: no cases, engine unreachable, or no trace for the request."""


class SftError(RuntimeError):
    """SFT cannot start or did not produce a GGUF."""
