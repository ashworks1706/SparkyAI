"""Running one live query: the checks, the fetch, and the text handed back to the engine."""

from __future__ import annotations

from scraper.core import telemetry
from scraper.core.settings import settings
from scraper.core.types import Job, QueryError, QueryResult, QuerySource
from scraper.ingest import extract, fetch
from scraper.ingest.pace import HostPacer
from scraper.query.params import check
from scraper.query.registry import QUERY_SOURCES


def source_of(job: Job) -> QuerySource:
    """The query source a job names. Raises QueryError for one that is not registered."""
    key = str(job.input.get("source", ""))
    source = QUERY_SOURCES.get(key)
    if source is None:
        raise QueryError(f"unknown source {key!r}; known: {', '.join(sorted(QUERY_SOURCES))}")
    return source


def url_for(source: QuerySource, params: dict[str, str]) -> str:
    """Checks the parameters, then builds the URL of a page source."""
    check(source, params)
    if source.to_url is None:
        raise QueryError(f"{source.key} is not fetched from one page")
    return source.to_url(params)


def offered(source: QuerySource) -> bool:
    """Whether source is served: one read only through the admin session needs auth.enabled."""
    return settings().auth.enabled or not (source.auth and source.answer is None)


def run(
    source: QuerySource, params: dict[str, str], pacer: HostPacer | None = None
) -> tuple[str, str]:
    """Checks the parameters and fetches the source. Returns the URL to cite and the text."""
    if not offered(source):
        raise QueryError(f"{source.key} is off: it reads through a login and auth.enabled is false")
    if source.answer is not None:
        check(source, params)
        return source.answer(params)
    url = url_for(source, params)
    if pacer is not None:
        pacer.wait(url)
    fetched = fetch.fetch(url, needs_js=source.needs_js, auth=source.auth)
    text = source.extractor(fetched) if source.extractor else extract.page_text(fetched)
    return url, text


def run_job(job: Job, pacer: HostPacer | None = None) -> QueryResult:
    """Fetches one live query, with the whole text it produced. Raises QueryError on a bad input."""
    source = source_of(job)
    raw = job.input.get("params")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise QueryError("params must be an object")
    params = {k: str(v) for k, v in raw.items() if v is not None}

    with telemetry.tracer().start_as_current_span(
        "scrape.query",
        attributes={"sparky.source": source.key},
    ) as span:
        url, text = run(source, params, pacer)
        span.set_attribute("sparky.input", url)
        text = text.strip()
        if not text:
            raise QueryError(f"{source.key} returned a page with no readable text")
        span.set_attribute("sparky.output", f"{len(text)} chars")
    return QueryResult(source=source.key, url=url, text=text)


def for_caller(text: str, limit: int) -> str:
    """The text handed back to the engine, held to limit characters."""
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def should_index(source: QuerySource, enabled: bool) -> bool:
    """Whether a result of source is written to the retrieval index."""
    return enabled and source.index
