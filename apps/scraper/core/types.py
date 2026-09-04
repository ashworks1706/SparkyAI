"""Data types shared across the scraper."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A registered public ASU source. A row in `sources`, never a folder."""

    key: str
    url: str
    category: str
    fetch_every_hours: int = 24
    needs_js: bool = False


@dataclass(frozen=True)
class Fetched:
    """One fetched page."""

    url: str
    status: int
    body: bytes
    content_type: str


@dataclass(frozen=True)
class SourceRow:
    """The `sources` row for a source."""

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
