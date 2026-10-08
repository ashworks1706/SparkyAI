"""Writes the retrieval index: sources, source_versions, chunks, the tree, and query_sources."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import datetime

import psycopg

from scraper.core.types import ChunkRow, QuerySource, SourceRow, StoreError, TreeNode


def upsert_source(
    conn: psycopg.Connection, key: str, url: str, category: str, fetch_every_hours: int
) -> SourceRow:
    """Creates or refreshes the sources row and stamps this run as an attempt."""
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


def set_source_title(conn: psycopg.Connection, source_id: uuid.UUID, title: str) -> None:
    """Records the page title of a source. It is what a citation of the source is labelled with."""
    title = title.strip()
    if not title:
        return
    conn.execute("update sources set title = %s where id = %s", (title, source_id))


def latest_version(conn: psycopg.Connection, source_id: uuid.UUID) -> dict | None:
    """Most recent source_versions row, or None."""
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
    """Records one indexed version, with counts the next run's quality floor compares against."""
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
    """Drops the previous chunks for the source and writes the new ones."""
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
    """Writes the summary levels above the leaves and points every covered row at its parent."""
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
    """Per-source: the schedule, the last attempt, the last change, and the counts."""
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


def prune_versions(conn: psycopg.Connection, keep: int, batch: int) -> list[str]:
    """Removes each source's versions past the newest keep, at most batch of them."""
    if keep <= 0 or batch <= 0:
        return []
    rows = conn.execute(
        """
        select id, snapshot_key from (
            select id, snapshot_key,
                   row_number() over (partition by source_id order by fetched_at desc) as rank
            from source_versions
        ) ranked
        where rank > %s
          and not exists (select 1 from chunks c where c.version_id = ranked.id)
        limit %s
        """,
        (keep, batch),
    ).fetchall()
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    conn.execute(
        "update source_versions set previous_id = null where previous_id = any(%s)", (ids,)
    )
    conn.execute("delete from source_versions where id = any(%s)", (ids,))
    return [r["snapshot_key"] for r in rows if r["snapshot_key"]]


def upsert_query_sources(conn: psycopg.Connection, sources: Sequence[QuerySource]) -> int:
    """Publishes the registry the engine checks its search tools against."""
    keys = [s.key for s in sources]
    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into query_sources (key, description, params, indexed, enabled, updated_at)
            values (%s, %s, %s::jsonb, %s, true, now())
            on conflict (key) do update
              set description = excluded.description,
                  params = excluded.params,
                  indexed = excluded.indexed,
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
                    s.index,
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


def _vector(values: Sequence[float]) -> str:
    """A pgvector text literal."""
    return "[" + ",".join(repr(float(x)) for x in values) + "]"
