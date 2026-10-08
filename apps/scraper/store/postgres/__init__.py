"""PostgreSQL access: the pool, the migrations runner, the retrieval index, and the jobs queue."""

from __future__ import annotations

from scraper.store.postgres.index import (
    insert_tree,
    insert_version,
    latest_version,
    leaf_ids,
    prune_versions,
    replace_chunks,
    set_source_title,
    status_rows,
    upsert_query_sources,
    upsert_source,
)
from scraper.store.postgres.jobs import (
    backlog_at_least,
    claim_job,
    enqueue_job,
    fail_job,
    finish_job,
    prune_jobs,
    queue_counts,
    requeue_stale,
)
from scraper.store.postgres.migrate import CONCURRENT_MARKER, MIGRATIONS_DIR, migrate, statements
from scraper.store.postgres.pool import connection, pool

__all__ = [
    "CONCURRENT_MARKER",
    "MIGRATIONS_DIR",
    "backlog_at_least",
    "claim_job",
    "connection",
    "enqueue_job",
    "fail_job",
    "finish_job",
    "insert_tree",
    "insert_version",
    "latest_version",
    "leaf_ids",
    "migrate",
    "pool",
    "prune_jobs",
    "prune_versions",
    "queue_counts",
    "replace_chunks",
    "requeue_stale",
    "set_source_title",
    "statements",
    "status_rows",
    "upsert_query_sources",
    "upsert_source",
]
