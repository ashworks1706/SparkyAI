"""Runs live query jobs the engine queues. The only online path in the scraper.

A query arrives as a jobs row and its answer goes back the same way. Nothing here writes to
chunks. A live result answers one caller and is not retrieval evidence.
"""

from __future__ import annotations

import time

import structlog

from scraper.core import telemetry
from scraper.core.settings import settings
from scraper.core.types import Job, QueryError, QueryResult
from scraper.ingest import extract, fetch
from scraper.query.registry import QUERY_SOURCES, url_for
from scraper.store import postgres

log = structlog.get_logger()

#: jobs.kind this worker claims.
KIND = "source_query"


def run_job(job: Job) -> QueryResult:
    """Fetches one live query. Raises QueryError with a reason the model can act on."""
    key = str(job.input.get("source", ""))
    source = QUERY_SOURCES.get(key)
    if source is None:
        raise QueryError(f"unknown source {key!r}; known: {', '.join(sorted(QUERY_SOURCES))}")
    # The fallback applies only when params is absent; the type is checked separately.
    raw = job.input.get("params")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise QueryError("params must be an object")
    params = {k: str(v) for k, v in raw.items() if v is not None}

    url = url_for(source, params)
    with telemetry.tracer().start_as_current_span(
        "scrape.query",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": url,
            "sparky.source": source.key,
        },
    ) as span:
        fetched = fetch.fetch(url, needs_js=source.needs_js)
        text = fetched.text if fetched.text is not None else extract.extract_text(fetched.body)
        text = text.strip()
        if not text:
            raise QueryError(f"{source.key} returned a page with no readable text")
        limit = settings().scraper.query_max_chars
        if len(text) > limit:
            text = text[:limit] + "\n…[truncated]"
        span.set_attribute("output.value", f"{len(text)} chars")
    return QueryResult(source=source.key, url=url, text=text)


def poll_once() -> bool:
    """Claims and runs at most one job. Returns whether there was one."""
    with postgres.connection() as conn:
        job = postgres.claim_job(conn, KIND)
        conn.commit()
        if job is None:
            return False
        try:
            result = run_job(job)
        except QueryError as e:
            # The reason is stored on the job.
            log.info("query rejected", job=str(job.id), error=str(e))
            postgres.fail_job(conn, job.id, str(e))
        except Exception as e:
            log.error("query failed", job=str(job.id), error=str(e))
            postgres.fail_job(conn, job.id, f"{type(e).__name__}: {e}")
        else:
            log.info("query answered", job=str(job.id), source=result.source)
            postgres.finish_job(
                conn, job.id, {"source": result.source, "url": result.url, "text": result.text}
            )
        conn.commit()
    return True


def serve(poll_secs: float = 0.5) -> None:
    """Publishes the registry, then runs jobs until stopped."""
    with postgres.connection() as conn:
        count = postgres.upsert_query_sources(conn, list(QUERY_SOURCES.values()))
        conn.commit()
    log.info("registry published", sources=count)
    while True:
        try:
            if poll_once():
                continue
        except Exception as e:
            # The next poll retries.
            log.error("worker poll failed", error=str(e))
        time.sleep(poll_secs)
