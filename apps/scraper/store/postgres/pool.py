"""The process-wide psycopg connection pool."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from scraper.core.settings import settings

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    """Process-wide pool, opened on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings().postgres.url.get_secret_value(),
            min_size=1,
            max_size=settings().postgres.scraper_pool_max,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """One pooled connection, committed on success."""
    with pool().connection() as conn:
        yield conn
