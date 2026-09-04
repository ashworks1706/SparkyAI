"""`scraper run <source>|--all`, `scraper schedule`, `scraper status`, `scraper migrate`."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import structlog
import typer

from scraper import pipeline
from scraper.core import telemetry
from scraper.sources import SOURCES
from scraper.store import postgres

app = typer.Typer(no_args_is_help=True)
log = structlog.get_logger()


@app.callback()
def main() -> None:
    """Scraper: keep the retrieval index fresh."""
    telemetry.init()
    structlog.configure(
        processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.dev.ConsoleRenderer()]
    )


@app.command()
def run(
    source: str | None = typer.Argument(None, help="Source key, e.g. library_hours."),
    all_sources: bool = typer.Option(False, "--all", help="Run every registered source."),
    force: bool = typer.Option(
        False, "--force", help="Re-index even if the page hash is unchanged."
    ),
) -> None:
    """Fetch, extract, chunk, embed, and index one source or every source."""
    if not source and not all_sources:
        raise typer.BadParameter("give a source key or --all")
    keys = list(SOURCES) if all_sources else [source]
    failures = 0
    for key in keys:
        src = SOURCES.get(key)
        if src is None:
            typer.echo(f"unknown source: {key}. Known: {', '.join(SOURCES)}", err=True)
            raise typer.Exit(2)
        try:
            result = pipeline.run_source(src, force=force)
            typer.echo(
                f"{key}: {'indexed' if result.changed else 'unchanged'} ({result.chunks} chunks)"
            )
        except Exception as e:  # noqa: BLE001 — one bad source must not stop the rest
            failures += 1
            log.error("source failed", source=key, error=str(e))
            typer.echo(f"{key}: FAILED — {e}", err=True)
    if failures:
        raise typer.Exit(1)


@app.command()
def schedule(poll_secs: int = typer.Option(300, help="How often to check what is due.")) -> None:
    """Runs each source when its `fetch_every` interval has elapsed. Blocks forever."""
    while True:
        with postgres.connection() as conn:
            rows = {r["key"]: r for r in postgres.status_rows(conn)}
        now = datetime.now(UTC)
        for key, src in SOURCES.items():
            last = rows.get(key, {}).get("last_fetch")
            due = last is None or now - last >= timedelta(hours=src.fetch_every_hours)
            if not due:
                continue
            try:
                pipeline.run_source(src)
            except Exception as e:  # noqa: BLE001
                log.error("scheduled run failed", source=key, error=str(e))
        time.sleep(poll_secs)


@app.command()
def status() -> None:
    """Last fetch time, version count, and chunk count per source."""
    with postgres.connection() as conn:
        rows = postgres.status_rows(conn)
    if not rows:
        typer.echo("no sources yet; run `scraper run --all`")
        return
    typer.echo(f"{'source':<16}{'category':<14}{'last fetch':<22}{'versions':>9}{'chunks':>8}")
    for r in rows:
        last = r["last_fetch"].strftime("%Y-%m-%d %H:%M UTC") if r["last_fetch"] else "-"
        typer.echo(f"{r['key']:<16}{r['category']:<14}{last:<22}{r['versions']:>9}{r['chunks']:>8}")


@app.command()
def migrate() -> None:
    """Apply `migrations/` to Postgres."""
    with postgres.connection() as conn:
        applied = postgres.migrate(conn)
    typer.echo("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
