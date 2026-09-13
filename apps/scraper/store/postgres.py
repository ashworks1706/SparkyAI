"""psycopg pool; migrations runner; writes sources, source_versions, chunks, and the tree."""

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
from scraper.core.types import ChunkRow, Job, QuerySource, SourceRow, StoreError, TreeNode

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
    """Applies every unapplied NNNN_*.sql in order. Returns the names applied."""
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
    """Creates or refreshes the sources row and stamps this run as an attempt.

    fetch_every is set on insert only. The scheduler reads the column.
    """
    row = conn.execute(
        """
        insert into sources (key, url, category, fetch_every, last_attempt_at)
        values (%s, %s, %s, make_interval(hours => %s), now())
        on conflict (key) do update
            set url = excluded.url, category = excluded.category, last_attempt_at = now()
        returning id, key, url, category
        """,
        (key, url, category, fetch_every_hours),
    ).fetchone()
    if row is None:
        raise StoreError(f"upsert of source {key!r} returned no row")
    return SourceRow(id=row["id"], key=row["key"], url=row["url"], category=row["category"])


def latest_version(conn: psycopg.Connection, source_id: uuid.UUID) -> dict | None:
    """Most recent source_versions row, or None. Only runs that pass the quality floor write a
    version."""
    return conn.execute(
        """
        select id, content_hash, fetched_at, text_chars, chunk_count from source_versions
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
    text_chars: int,
    chunk_count: int,
) -> uuid.UUID:
    """Records one indexed version. text_chars and chunk_count are what the next run's quality
    floor compares against."""
    row = conn.execute(
        """
        insert into source_versions
            (source_id, content_hash, snapshot_key, parser_version, chunker_version,
             embedding_model, previous_id, text_chars, chunk_count)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            text_chars,
            chunk_count,
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
    """Drops the previous chunks for the source and writes the new ones. The index reflects the
    current page; source_versions keeps the history."""
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
                    _vector(c.embedding),
                    fetched_at,
                )
                for c in chunks
            ],
        )
    return len(chunks)


def leaf_ids(conn: psycopg.Connection, version_id: uuid.UUID) -> list[uuid.UUID]:
    """Ids of one version's leaf chunks, in ordinal order."""
    rows = conn.execute(
        "select id from chunks where version_id = %s and level = 0 order by ordinal",
        (version_id,),
    ).fetchall()
    return [r["id"] for r in rows]


def insert_tree(
    conn: psycopg.Connection,
    *,
    tenant_id: str,
    source: SourceRow,
    version_id: uuid.UUID,
    fetched_at: datetime,
    leaves: Sequence[uuid.UUID],
    nodes: Sequence[TreeNode],
) -> int:
    """Writes the summary levels above the leaves and points every covered row at its parent.

    nodes come parents last. Every child holds an id before its parent is written.
    """
    ids = list(leaves)
    for node in nodes:
        row = conn.execute(
            """
            insert into chunks
                (tenant_id, source_id, version_id, category, ordinal, content, embedding,
                 fetched_at, level)
            values (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
            returning id
            """,
            (
                tenant_id,
                source.id,
                version_id,
                source.category,
                node.ordinal,
                node.content,
                _vector(node.embedding),
                fetched_at,
                node.level,
            ),
        ).fetchone()
        if row is None:
            raise StoreError(f"insert of tree node {node.ordinal} returned no row")
        conn.execute(
            "update chunks set parent_id = %s where id = any(%s)",
            (row["id"], [ids[position] for position in node.children]),
        )
        ids.append(row["id"])
    return len(nodes)


def status_rows(conn: psycopg.Connection) -> list[dict]:
    """Per-source: the schedule, the last attempt, the last change, and the counts.

    last_attempt is when a run last touched the source; last_fetch is when one last produced a
    new version. The two differ whenever the page came back unchanged.
    """
    return conn.execute(
        """
        select s.key, s.category, s.enabled, s.fetch_every,
               s.last_attempt_at as last_attempt,
               (select max(fetched_at) from source_versions v where v.source_id = s.id)
                   as last_fetch,
               (select count(*) from source_versions v where v.source_id = s.id) as versions,
               (select count(*) from chunks c where c.source_id = s.id) as chunks
        from sources s order by s.key
        """
    ).fetchall()


def upsert_query_sources(conn: psycopg.Connection, sources: Sequence[QuerySource]) -> int:
    """Publishes the registry the engine checks its search tools against. Sources no longer in
    code are disabled, not deleted."""
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
                                "choices": list(p.choices),
                                "many": p.many,
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


def claim_job(conn: psycopg.Connection, kinds: Sequence[str]) -> Job | None:
    """Takes the queued job of one of kinds with the highest priority, oldest first, or None.
    skip locked lets several lanes and processes claim concurrently."""
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


def requeue_stale(conn: psycopg.Connection, kinds: Sequence[str], lease_secs: float) -> int:
    """Puts back jobs of kinds left running longer than lease_secs, as a stopped process leaves
    them. Returns how many."""
    cursor = conn.execute(
        """
        update jobs set status = 'queued', updated_at = now()
        where status = 'running' and kind = any(%s)
          and updated_at < now() - make_interval(secs => %s)
        """,
        (list(kinds), lease_secs),
    )
    return cursor.rowcount


def _vector(values: Sequence[float]) -> str:
    """A pgvector text literal."""
    return "[" + ",".join(repr(float(x)) for x in values) + "]"


def queue_counts(conn: psycopg.Connection) -> list[dict]:
    """Jobs by kind and status."""
    return conn.execute(
        """
        select kind, status, count(*) as jobs from jobs
        group by kind, status order by kind, status
        """
    ).fetchall()
