"""psycopg pool; migrations runner; writes sources, source_versions, and chunks."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from scraper.core.settings import settings
from scraper.core.types import ChunkRow, Job, QuerySource, SourceRow, StoreError

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    """Process-wide pool, opened on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings().postgres.url.get_secret_value(),
            min_size=1,
            max_size=4,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """One pooled connection, committed on success."""
    with pool().connection() as conn:
        yield conn


def migrate(conn: psycopg.Connection) -> list[str]:
    """Applies every unapplied `NNNN_*.sql` in order. Returns the names applied."""
    conn.execute(
        """
        create table if not exists schema_migrations (
            name text primary key,
            applied_at timestamptz not null default now()
        )
        """
    )
    applied = {r["name"] for r in conn.execute("select name from schema_migrations").fetchall()}
    done: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
        done.append(path.name)
    conn.commit()
    return done


def upsert_source(
    conn: psycopg.Connection, key: str, url: str, category: str, fetch_every_hours: int
) -> SourceRow:
    """Creates or refreshes the `sources` row for a registered source."""
    row = conn.execute(
        """
        insert into sources (key, url, category, fetch_every)
        values (%s, %s, %s, make_interval(hours => %s))
        on conflict (key) do update
            set url = excluded.url, category = excluded.category, fetch_every = excluded.fetch_every
        returning id, key, url, category
        """,
        (key, url, category, fetch_every_hours),
    ).fetchone()
    if row is None:
        raise StoreError(f"upsert of source {key!r} returned no row")
    return SourceRow(id=row["id"], key=row["key"], url=row["url"], category=row["category"])


def latest_version(conn: psycopg.Connection, source_id: uuid.UUID) -> dict | None:
    """Most recent `source_versions` row, or None."""
    return conn.execute(
        """
        select id, content_hash, fetched_at from source_versions
        where source_id = %s order by fetched_at desc limit 1
        """,
        (source_id,),
    ).fetchone()


def insert_version(
    conn: psycopg.Connection,
    *,
    source_id: uuid.UUID,
    content_hash: str,
    snapshot_key: str,
    parser_version: str,
    chunker_version: str,
    embedding_model: str,
    previous_id: uuid.UUID | None,
) -> uuid.UUID:
    row = conn.execute(
        """
        insert into source_versions
            (source_id, content_hash, snapshot_key, parser_version, chunker_version,
             embedding_model, previous_id)
        values (%s, %s, %s, %s, %s, %s, %s)
        returning id
        """,
        (
            source_id,
            content_hash,
            snapshot_key,
            parser_version,
            chunker_version,
            embedding_model,
            previous_id,
        ),
    ).fetchone()
    if row is None:
        raise StoreError(f"insert of version for source {source_id} returned no row")
    return row["id"]


def replace_chunks(
    conn: psycopg.Connection,
    *,
    tenant_id: str,
    source: SourceRow,
    version_id: uuid.UUID,
    fetched_at: datetime,
    chunks: Sequence[ChunkRow],
) -> int:
    """Drops the source's previous chunks and writes the new version's. The index reflects the
    current page; `source_versions` keeps the history."""
    conn.execute("delete from chunks where source_id = %s", (source.id,))
    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into chunks
                (tenant_id, source_id, version_id, category, ordinal, content, embedding,
                 fetched_at)
            values (%s, %s, %s, %s, %s, %s, %s::vector, %s)
            """,
            [
                (
                    tenant_id,
                    source.id,
                    version_id,
                    source.category,
                    c.ordinal,
                    c.content,
                    "[" + ",".join(repr(float(x)) for x in c.embedding) + "]",
                    fetched_at,
                )
                for c in chunks
            ],
        )
    return len(chunks)


def status_rows(conn: psycopg.Connection) -> list[dict]:
    """Per-source: last fetch, version count, chunk count."""
    return conn.execute(
        """
        select s.key, s.category, s.enabled,
               (select max(fetched_at) from source_versions v where v.source_id = s.id)
                   as last_fetch,
               (select count(*) from source_versions v where v.source_id = s.id) as versions,
               (select count(*) from chunks c where c.source_id = s.id) as chunks
        from sources s order by s.key
        """
    ).fetchall()


def upsert_query_sources(conn: psycopg.Connection, sources: Sequence[QuerySource]) -> int:
    """Publishes the registry the engine reads to build its tool. Sources no longer in code are
    disabled rather than deleted, so a job already queued against one still resolves."""
    keys = [s.key for s in sources]
    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into query_sources (key, description, params, enabled, updated_at)
            values (%s, %s, %s::jsonb, true, now())
            on conflict (key) do update
              set description = excluded.description,
                  params = excluded.params,
                  enabled = true,
                  updated_at = now()
            """,
            [
                (
                    s.key,
                    s.description,
                    json.dumps(
                        [
                            {
                                "name": p.name,
                                "description": p.description,
                                "required": p.required,
                                "example": p.example,
                            }
                            for p in s.params
                        ]
                    ),
                )
                for s in sources
            ],
        )
        cur.execute(
            "update query_sources set enabled = false, updated_at = now() "
            "where enabled and not (key = any(%s))",
            (keys,),
        )
    return len(keys)


def claim_job(conn: psycopg.Connection, kind: str) -> Job | None:
    """Takes the oldest queued job of `kind`, or `None`. `skip locked` lets several workers run
    without one blocking on another's row."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            update jobs set status = 'running', attempts = attempts + 1, updated_at = now()
            where id = (
                select id from jobs
                where status = 'queued' and kind = %s
                  and (deadline is null or deadline > now())
                order by created_at
                for update skip locked
                limit 1
            )
            returning id, kind, input
            """,
            (kind,),
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
