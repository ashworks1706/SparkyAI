"""The migrations runner: applies apps/scraper/migrations in order and records each one."""

from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

#: First line of a migration that runs one statement at a time outside a transaction.
CONCURRENT_MARKER = "-- concurrent:"


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
        sql = path.read_text(encoding="utf-8")
        if sql.lstrip().startswith(CONCURRENT_MARKER):
            _apply_unwrapped(conn, sql)
        else:
            conn.execute(sql)
        conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
        done.append(path.name)
    conn.commit()
    return done


def statements(sql: str) -> list[str]:
    """Splits SQL on the semicolons that end a statement, ignoring quotes and line comments."""
    out: list[str] = []
    current: list[str] = []
    quoted = False
    commented = False
    previous = ""
    for ch in sql:
        if commented:
            current.append(ch)
            if ch == "\n":
                commented = False
        elif quoted:
            current.append(ch)
            if ch == "'":
                quoted = False
        elif ch == "'":
            quoted = True
            current.append(ch)
        elif ch == "-" and previous == "-":
            commented = True
            current.append(ch)
        elif ch == ";":
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
        previous = ch
    out.append("".join(current))
    return [s for s in (part.strip() for part in out) if s and not _only_comments(s)]


def _only_comments(statement: str) -> bool:
    """Whether a statement is nothing but comment and blank lines."""
    return all(not line.strip() or line.strip().startswith("--") for line in statement.splitlines())


def _apply_unwrapped(conn: psycopg.Connection, sql: str) -> None:
    """Runs a migration one statement at a time outside a transaction."""
    conn.commit()
    was = conn.autocommit
    conn.autocommit = True
    try:
        for statement in statements(sql):
            conn.execute(statement)
    finally:
        conn.autocommit = was
