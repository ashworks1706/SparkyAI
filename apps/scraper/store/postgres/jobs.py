"""The jobs table: claims, finishes, fails, enqueues, requeues, prunes, and counts jobs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence

import psycopg
from psycopg.rows import dict_row

from scraper.core.types import Job


def claim_job(conn: psycopg.Connection, kinds: Sequence[str]) -> Job | None:
    """Takes the queued job of one of kinds with the highest priority, oldest first, or None."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            update jobs set status = 'running', attempts = attempts + 1, updated_at = now()
            where id = (
                select id from jobs
                where status = 'queued' and kind = any(%s)
                  and (deadline is null or deadline > now())
                order by priority desc, created_at
                for update skip locked
                limit 1
            )
            returning id, kind, input
            """,
            (list(kinds),),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Job(id=row["id"], kind=row["kind"], input=row["input"] or {})


def finish_job(conn: psycopg.Connection, job_id: uuid.UUID, result: dict) -> None:
    """Marks a job done and stores what it produced."""
    conn.execute(
        "update jobs set status = 'done', result = %s::jsonb, updated_at = now() where id = %s",
        (json.dumps(result), job_id),
    )


def fail_job(conn: psycopg.Connection, job_id: uuid.UUID, error: str) -> None:
    """Marks a job failed with a reason the caller can read."""
    conn.execute(
        "update jobs set status = 'failed', error = %s, updated_at = now() where id = %s",
        (error[:2000], job_id),
    )


def enqueue_job(conn: psycopg.Connection, kind: str, input: dict, priority: int) -> bool:
    """Queues a job. Returns False when a unique queue rule already holds an equal one."""
    row = conn.execute(
        """
        insert into jobs (kind, status, input, priority)
        values (%s, 'queued', %s::jsonb, %s)
        on conflict do nothing
        returning id
        """,
        (kind, json.dumps(input), priority),
    ).fetchone()
    return row is not None


def backlog_at_least(conn: psycopg.Connection, kind: str, limit: int) -> bool:
    """Whether at least limit jobs of kind are queued. Reads no further than limit rows."""
    if limit <= 0:
        return False
    row = conn.execute(
        "select 1 from jobs where kind = %s and status = 'queued' offset %s limit 1",
        (kind, limit - 1),
    ).fetchone()
    return row is not None


def requeue_stale(conn: psycopg.Connection, kinds: Sequence[str], lease_secs: float) -> int:
    """Puts back jobs of kinds left running longer than lease_secs. Returns how many."""
    cursor = conn.execute(
        """
        update jobs set status = 'queued', updated_at = now()
        where status = 'running' and kind = any(%s)
          and updated_at < now() - make_interval(secs => %s)
        """,
        (list(kinds), lease_secs),
    )
    return cursor.rowcount


def prune_jobs(conn: psycopg.Connection, older_than_secs: float, batch: int) -> int:
    """Removes finished jobs older than older_than_secs, at most batch of them. Returns how many."""
    if older_than_secs <= 0 or batch <= 0:
        return 0
    cursor = conn.execute(
        """
        delete from jobs where id in (
            select id from jobs
            where status in ('done', 'failed', 'cancelled')
              and updated_at < now() - make_interval(secs => %s)
            limit %s
        )
        """,
        (older_than_secs, batch),
    )
    return cursor.rowcount


def queue_counts(conn: psycopg.Connection) -> list[dict]:
    """Jobs by kind and status."""
    return conn.execute(
        """
        select kind, status, count(*) as jobs from jobs
        group by kind, status order by kind, status
        """
    ).fetchall()
