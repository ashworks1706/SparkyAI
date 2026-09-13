"""scraper run <source>|--all, scraper serve, scraper status, scraper migrate."""

from __future__ import annotations

from datetime import datetime

import structlog
import typer

from scraper import jobs
from scraper.core import telemetry
from scraper.ingest import pipeline
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
        False,
        "--force",
        help="Re-index even if the page hash is unchanged or the page shrank below the floor.",
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
        except Exception as e:  # one bad source does not stop the rest
            failures += 1
            log.error("source failed", source=key, error=str(e))
            typer.echo(f"{key}: FAILED — {e}", err=True)
    if failures:
        raise typer.Exit(1)


@app.command()
def serve() -> None:
    """Run the scraper: live queries, result indexing, scheduled sources, one job queue. Blocks."""
    jobs.serve()


@app.command()
def status() -> None:
    """Last attempt, last change, version count, and chunk count per source, then the queue."""
    with postgres.connection() as conn:
        rows = postgres.status_rows(conn)
        queue = postgres.queue_counts(conn)
    if not rows:
        typer.echo("no sources yet; run `scraper run --all`")
        return
    typer.echo(
        f"{'source':<16}{'category':<14}{'last attempt':<22}{'last change':<22}"
        f"{'versions':>9}{'chunks':>8}"
    )
    for r in rows:
        attempt = _stamp(r["last_attempt"])
        change = _stamp(r["last_fetch"])
        typer.echo(
            f"{r['key']:<16}{r['category']:<14}{attempt:<22}{change:<22}"
            f"{r['versions']:>9}{r['chunks']:>8}"
        )

    if queue:
        typer.echo(f"\n{'job kind':<16}{'status':<14}{'jobs':>8}")
        for q in queue:
            typer.echo(f"{q['kind']:<16}{q['status']:<14}{q['jobs']:>8}")


def _stamp(at: datetime | None) -> str:
    return at.strftime("%Y-%m-%d %H:%M UTC") if at else "-"


@app.command()
def migrate() -> None:
    """Apply migrations/ to Postgres."""
    with postgres.connection() as conn:
        applied = postgres.migrate(conn)
    typer.echo("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
