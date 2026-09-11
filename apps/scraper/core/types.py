"""Data types shared across the scraper."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A registered public ASU source. A row in sources, never a folder.

    extractor, when set, turns the fetched page into clean text in place of the shared
    heuristic in extract.py.
    """

    key: str
    url: str
    category: str
    fetch_every_hours: int = 24
    needs_js: bool = False
    extractor: Callable[[Fetched], str] | None = None


@dataclass(frozen=True)
class QueryParam:
    """One parameter a query source accepts. Shown to the model, checked by the worker."""

    name: str
    description: str
    required: bool = False
    example: str | None = None


@dataclass(frozen=True)
class QuerySource:
    """A source the model queries live with parameters. Not fetched on a schedule.

    to_url turns the model parameters into the URL to fetch. Results answer one caller and
    are never written to the retrieval index.
    """

    key: str
    description: str
    params: tuple[QueryParam, ...]
    to_url: Callable[[dict[str, str]], str]
    needs_js: bool = False


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


class PipelineError(RuntimeError):
    """A source run cannot produce a trustworthy index."""


class StoreError(RuntimeError):
    """Postgres returned something other than what the statement promised."""
