"""Data types shared across the scraper."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A registered public ASU source. A row in sources, never a folder.

    extractor, when set, turns the fetched page into clean text in place of the shared
    heuristic in ingest/extract.py.
    """

    key: str
    url: str
    category: str
    fetch_every_hours: int = 24
    needs_js: bool = False
    extractor: Callable[[Fetched], str] | None = None


@dataclass(frozen=True)
class QueryParam:
    """One parameter a query source accepts. Published to the engine, checked by the worker.

    choices, when set, are the only values accepted, compared without case. many accepts a
    comma-separated list of them.
    """

    name: str
    description: str
    required: bool = False
    example: str | None = None
    choices: tuple[str, ...] = ()
    many: bool = False


@dataclass(frozen=True)
class QuerySource:
    """A source the model queries live with parameters. Not fetched on a schedule.

    Exactly one of to_url and answer is set. to_url turns the model parameters into the one
    page to fetch, and extractor, when set, turns that page into clean text in place of the
    shared heuristic. answer does the fetching itself, for a source that reads several
    endpoints, and returns the URL to cite with the text. Results answer one caller and are
    never written to the retrieval index.
    """

    key: str
    description: str
    params: tuple[QueryParam, ...]
    to_url: Callable[[dict[str, str]], str] | None = None
    needs_js: bool = False
    extractor: Callable[[Fetched], str] | None = None
    answer: Callable[[dict[str, str]], tuple[str, str]] | None = None

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
    """One fetched page. text is set when the fetcher produced clean text (markdown from
    Firecrawl). When text is None the pipeline extracts it from body."""

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
    """One summary node above the leaves, ready to write.

    children are positions in the node list the tree was built from: 0 to leaves - 1 are the
    leaf chunks in ordinal order, and everything after them is a summary node in build order.
    A node always comes after the nodes it covers.
    """

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
