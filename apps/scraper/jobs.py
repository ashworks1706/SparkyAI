"""The scraper job queue: job kinds, the lanes that claim them, and the scheduled-run timer."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime

import psycopg
import structlog

from scraper.core.settings import settings
from scraper.core.types import Job, QueryError
from scraper.ingest import pipeline
from scraper.ingest.pace import HostPacer
from scraper.query import index
from scraper.query.registry import QUERY_SOURCES
from scraper.query.run import for_caller, run_job, should_index, source_of
from scraper.sources import SOURCES
from scraper.store import object as objects
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
    """Whether a source should run, from its sources row. No row means never attempted, so due."""
    if row is None:
        return True
    if not row["enabled"]:
        return False
    last = row["last_attempt"]
    if last is None:
        return True
    return now - last >= row["fetch_every"]


def handle(conn: psycopg.Connection, job: Job, pacer: HostPacer | None = None) -> dict:
    """Does one job and returns its result. Raises QueryError for a job the caller can correct."""
    cfg = settings().scraper
    if job.kind == QUERY:
        result = run_job(job)
        if should_index(QUERY_SOURCES[result.source], cfg.index_live_results):
            # A live result is not queued for indexing past index_backlog_limit.
            if postgres.backlog_at_least(conn, INDEX, cfg.index_backlog_limit):
                log.warning("index backlog full; live result not queued", source=result.source)
            else:
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
        run = pipeline.run_source(SOURCES[key], pacer=pacer)
        return {"source": run.source, "changed": run.changed, "chunks": run.chunks}
    raise QueryError(f"no handler for job kind {job.kind!r}")


def poll_once(kinds: Sequence[str], pacer: HostPacer | None = None) -> bool:
    """Claims and does at most one job of kinds. Returns whether there was one."""
    with postgres.connection() as conn:
        job = postgres.claim_job(conn, kinds)
        conn.commit()
        if job is None:
            return False
        try:
            result = handle(conn, job, pacer)
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
    """Queues a run for every due source; requeues stale jobs and removes finished ones."""
    cfg = settings().scraper
    with postgres.connection() as conn:
        stale = postgres.requeue_stale(conn, BACKGROUND_KINDS, cfg.job_lease_secs)
        pruned = postgres.prune_jobs(conn, cfg.job_retention_hours * 3600.0, cfg.job_prune_batch)
        snapshots = postgres.prune_versions(conn, cfg.keep_versions, cfg.job_prune_batch)
        rows = {r["key"]: r for r in postgres.status_rows(conn)}
        queued = sum(
            postgres.enqueue_job(conn, RUN, {"source": key}, RUN_PRIORITY)
            for key in SOURCES
            if is_due(rows.get(key), now)
        )
        conn.commit()
    if snapshots:
        try:
            objects.delete_snapshots(snapshots)
        except Exception as e:  # the rows are gone; an unreachable store leaves only objects
            log.warning("old snapshots not removed", count=len(snapshots), error=str(e))
    if stale or queued or pruned or snapshots:
        log.info(
            "queue topped up",
            runs=queued,
            requeued=stale,
            pruned=pruned,
            versions_removed=len(snapshots),
        )
    return queued


def _lane(name: str, kinds: Sequence[str], stop: threading.Event, listen: bool) -> None:
    """Claims jobs of kinds until stop is set. A listening lane also wakes on notification."""
    cfg = settings()
    pacer = HostPacer(cfg.scraper.host_gap_secs)
    listener = None
    if listen:
        listener = psycopg.connect(cfg.postgres.url.get_secret_value(), autocommit=True)
        listener.execute(f"listen {CHANNEL}")
    try:
        while not stop.is_set():
            try:
                if poll_once(kinds, pacer):
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
    """Publishes the query registry, starts the live and background lanes, and schedules runs."""
    cfg = settings().scraper
    with postgres.connection() as conn:
        count = postgres.upsert_query_sources(conn, list(QUERY_SOURCES.values()))
        conn.commit()
    log.info("registry published", sources=count)
    stop = threading.Event()
    lanes = [(f"live-{i}", LIVE_KINDS, True) for i in range(1, cfg.live_workers + 1)]
    lanes.append(("background", BACKGROUND_KINDS, False))
    for name, kinds, listen in lanes:
        threading.Thread(
            target=_lane, args=(name, kinds, stop, listen), name=name, daemon=True
        ).start()
    log.info("serving", live=LIVE_KINDS, live_workers=cfg.live_workers, background=BACKGROUND_KINDS)
    try:
        while True:
            try:
                enqueue_due(datetime.now(UTC))
            except Exception as e:
                log.error("scheduling failed", error=str(e))
            stop.wait(cfg.schedule_every_secs)
    finally:
        stop.set()
