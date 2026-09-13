"""The scraper job queue: what each kind of job does, the lanes that claim them, and the timer
that queues scheduled runs.

Live queries from the engine run in their own lane, woken by the engine notification. Indexing a
live result and running a scheduled source share the background lane, highest priority first.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime

import psycopg
import structlog

from scraper.core.settings import settings
from scraper.core.types import Job, QueryError
from scraper.ingest import pipeline
from scraper.query import index
from scraper.query.registry import QUERY_SOURCES
from scraper.query.run import for_caller, run_job, should_index, source_of
from scraper.sources import SOURCES
from scraper.store import postgres

log = structlog.get_logger()

#: A live query the engine queued. The contract with the engine.
QUERY = "source_query"
#: Indexing the page a live query fetched.
INDEX = "live_index"
#: A scheduled run of a registered source.
RUN = "source_run"

#: The channel the engine notifies when it queues a live query.
CHANNEL = "source_query"

#: Priority of indexing a live result. The engine queues live queries at 0.
INDEX_PRIORITY = -10
#: Priority of a scheduled run.
RUN_PRIORITY = -20

#: Kinds the live lane claims.
LIVE_KINDS = (QUERY,)
#: Kinds the background lane claims.
BACKGROUND_KINDS = (INDEX, RUN)


def is_due(row: dict | None, now: datetime) -> bool:
    """Whether a source should run, from its sources row. A source with no row yet has never
    been attempted and is due; the first run creates the row from the registered source."""
    if row is None:
        return True
    if not row["enabled"]:
        return False
    last = row["last_attempt"]
    if last is None:
        return True
    return now - last >= row["fetch_every"]


def handle(conn: psycopg.Connection, job: Job) -> dict:
    """Does one job and returns its result. A live query that allows indexing queues the
    indexing of its page on conn, committed with the answer. Raises QueryError for a job the
    caller can correct."""
    cfg = settings().scraper
    if job.kind == QUERY:
        result = run_job(job)
        if should_index(QUERY_SOURCES[result.source], cfg.index_live_results):
            postgres.enqueue_job(
                conn,
                INDEX,
                {"source": result.source, "url": result.url, "text": result.text},
                INDEX_PRIORITY,
            )
        return {
            "source": result.source,
            "url": result.url,
            "text": for_caller(result.text, cfg.query_max_chars),
        }
    if job.kind == INDEX:
        query = source_of(job)
        run = index.index_result(query, str(job.input["url"]), str(job.input["text"]))
        return {"source": run.source, "changed": run.changed, "chunks": run.chunks}
    if job.kind == RUN:
        key = str(job.input.get("source", ""))
        if key not in SOURCES:
            raise QueryError(f"unknown source {key!r}")
        run = pipeline.run_source(SOURCES[key])
        return {"source": run.source, "changed": run.changed, "chunks": run.chunks}
    raise QueryError(f"no handler for job kind {job.kind!r}")


def poll_once(kinds: Sequence[str]) -> bool:
    """Claims and does at most one job of kinds. Returns whether there was one."""
    with postgres.connection() as conn:
        job = postgres.claim_job(conn, kinds)
        conn.commit()
        if job is None:
            return False
        try:
            result = handle(conn, job)
        except QueryError as e:
            conn.rollback()
            log.info("job rejected", job=str(job.id), kind=job.kind, error=str(e))
            postgres.fail_job(conn, job.id, str(e))
        except Exception as e:
            conn.rollback()
            log.error("job failed", job=str(job.id), kind=job.kind, error=str(e))
            postgres.fail_job(conn, job.id, f"{type(e).__name__}: {e}")
        else:
            log.info("job done", job=str(job.id), kind=job.kind, source=result.get("source"))
            postgres.finish_job(conn, job.id, result)
        conn.commit()
    return True


def enqueue_due(now: datetime) -> int:
    """Queues a run of every registered source that is due, and puts back background jobs a
    stopped process left running. Returns how many runs were queued."""
    cfg = settings().scraper
    with postgres.connection() as conn:
        stale = postgres.requeue_stale(conn, BACKGROUND_KINDS, cfg.job_lease_secs)
        rows = {r["key"]: r for r in postgres.status_rows(conn)}
        queued = sum(
            postgres.enqueue_job(conn, RUN, {"source": key}, RUN_PRIORITY)
            for key in SOURCES
            if is_due(rows.get(key), now)
        )
        conn.commit()
    if stale or queued:
        log.info("queue topped up", runs=queued, requeued=stale)
    return queued


def _lane(name: str, kinds: Sequence[str], stop: threading.Event, listen: bool) -> None:
    """Claims jobs of kinds until stop is set. A listening lane wakes on the engine
    notification; any lane looks again after poll_secs."""
    cfg = settings()
    listener = None
    if listen:
        listener = psycopg.connect(cfg.postgres.url.get_secret_value(), autocommit=True)
        listener.execute(f"listen {CHANNEL}")
    try:
        while not stop.is_set():
            try:
                if poll_once(kinds):
                    continue
            except Exception as e:
                # The next poll retries.
                log.error("lane poll failed", lane=name, error=str(e))
            if listener is not None:
                for _ in listener.notifies(timeout=cfg.scraper.serve_poll_secs, stop_after=1):
                    pass
            else:
                stop.wait(cfg.scraper.serve_poll_secs)
    finally:
        if listener is not None:
            listener.close()


def serve() -> None:
    """Publishes the query registry, starts the live and background lanes, and queues scheduled
    runs as they fall due. Blocks until interrupted."""
    cfg = settings().scraper
    with postgres.connection() as conn:
        count = postgres.upsert_query_sources(conn, list(QUERY_SOURCES.values()))
        conn.commit()
    log.info("registry published", sources=count)
    stop = threading.Event()
    for name, kinds, listen in (
        ("live", LIVE_KINDS, True),
        ("background", BACKGROUND_KINDS, False),
    ):
        threading.Thread(
            target=_lane, args=(name, kinds, stop, listen), name=name, daemon=True
        ).start()
    log.info("serving", live=LIVE_KINDS, background=BACKGROUND_KINDS)
    try:
        while True:
            try:
                enqueue_due(datetime.now(UTC))
            except Exception as e:
                log.error("scheduling failed", error=str(e))
            stop.wait(cfg.schedule_every_secs)
    finally:
        stop.set()
