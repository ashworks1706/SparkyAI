"""Writing what a live query fetched into the retrieval index, once its caller has the answer."""

from __future__ import annotations

import hashlib

import structlog

from scraper.core.types import Fetched, QuerySource, RunResult, Source
from scraper.ingest import pipeline
from scraper.sources import SOURCES

log = structlog.get_logger()

#: Hex characters of the URL digest a live page source key carries.
_DIGEST_CHARS = 10

#: The fetch interval stored for a live page source. The scheduler does not run it.
_LIVE_FETCH_EVERY_HOURS = 24


def page_source(query: QuerySource, url: str) -> Source:
    """The source a live page is indexed under."""
    scheduled = next((s for s in SOURCES.values() if s.url == url), None)
    if scheduled is not None:
        return Source(
            key=scheduled.key,
            url=scheduled.url,
            category=scheduled.category,
            fetch_every_hours=scheduled.fetch_every_hours,
        )
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return Source(
        key=f"{query.key}-{digest}",
        url=url,
        category=query.category,
        fetch_every_hours=_LIVE_FETCH_EVERY_HOURS,
    )


def live_page(query: QuerySource, url: str, text: str) -> Fetched:
    """The page a live query produced, as text already extracted."""
    return Fetched(
        url=url,
        status=200,
        body=text.encode("utf-8"),
        content_type="text/markdown",
        title=f"{query.key.replace('_', ' ')}: {url}",
        text=text,
    )


def index_result(query: QuerySource, url: str, text: str) -> RunResult:
    """Indexes one live result under the source page_source names."""
    source = page_source(query, url)
    result = pipeline.index_page(source, live_page(query, url, text))
    log.info(
        "live result indexed" if result.changed else "live result unchanged",
        source=source.key,
        chunks=result.chunks,
    )
    return result
