"""Data types shared across the scraper."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A public ASU source, a row in sources; extractor can replace the extraction heuristic."""

    key: str
    url: str
    category: str
    fetch_every_hours: int = 24
    needs_js: bool = False
    extractor: Callable[[Fetched], str] | None = None


@dataclass(frozen=True)
class QueryParam:
    """One param a query source takes; sent to engine, checked by worker. choices restrict input."""

    name: str
    description: str
    required: bool = False
    example: str | None = None
    choices: tuple[str, ...] = ()
    many: bool = False


@dataclass(frozen=True)
class QuerySource:
    """A source queried live with parameters, not scheduled. One of to_url or answer is set."""

    key: str
    description: str
    params: tuple[QueryParam, ...]
    to_url: Callable[[dict[str, str]], str] | None = None
    needs_js: bool = False
    extractor: Callable[[Fetched], str] | None = None
    answer: Callable[[dict[str, str]], tuple[str, str]] | None = None
    category: str = "live"
    index: bool = True

    def __post_init__(self) -> None:
        if (self.to_url is None) == (self.answer is None):
            raise ValueError(f"query source {self.key} sets exactly one of to_url and answer")


@dataclass(frozen=True)
class QueryResult:
    """What one live query produced."""

    source: str
    url: str
    text: str


class QueryError(RuntimeError):
    """The query could not be answered; the reason goes back to the model."""


@dataclass(frozen=True)
class Job:
    """A claimed jobs row."""

    id: uuid.UUID
    kind: str
    input: dict


@dataclass(frozen=True)
class Fetched:
    """One fetched page. text set if fetcher gave clean text; else pipeline extracts from body."""

    url: str
    status: int
    body: bytes
    content_type: str
    title: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class SourceRow:
    """The sources row for a source."""

    id: uuid.UUID
    key: str
    url: str
    category: str


@dataclass(frozen=True)
class ChunkRow:
    """One chunk ready to write."""

    ordinal: int
    content: str
    embedding: Sequence[float]


@dataclass(frozen=True)
class TreeNode:
    """One summary node above leaves, ready to write. children index positions in the built list."""

    level: int
    ordinal: int
    content: str
    embedding: Sequence[float]
    children: tuple[int, ...]


@dataclass(frozen=True)
class RunResult:
    """Outcome of one source run."""

    source: str
    changed: bool
    chunks: int
    content_hash: str


class FetchError(RuntimeError):
    """Transient fetch failure; retried."""


class FetchRejected(RuntimeError):
    """4xx from the origin; not retried."""


class EmbedError(RuntimeError):
    """The embedding endpoint refused or returned the wrong shape."""


class SummaryError(RuntimeError):
    """The chat endpoint refused, or answered a cluster with no summary."""


class PipelineError(RuntimeError):
    """A source run cannot produce a trustworthy index."""


class StoreError(RuntimeError):
    """Postgres returned something other than what the statement promised."""
