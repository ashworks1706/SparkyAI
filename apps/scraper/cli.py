"""scraper run <source>|--all, scraper serve, scraper status, scraper login, scraper migrate."""

from __future__ import annotations

import sys
from datetime import datetime

import structlog
import typer

from scraper import jobs
from scraper.core import telemetry
from scraper.core.settings import settings
from scraper.core.types import AuthError, FetchError
from scraper.ingest import pipeline
from scraper.ingest.drivers import admin
from scraper.ingest.drivers.asu_sso import Credentials
from scraper.ingest.pace import HostPacer
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
    category: str | None = typer.Option(
        None, "--category", help="Run every source in one category, e.g. housing."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-index even if the page hash is unchanged or the page shrank below the floor.",
    ),
) -> None:
    """Fetch, extract, chunk, embed, and index one source, a category, or every source."""
    keys = selected(source, all_sources=all_sources, category=category)
    require_session()
    pacer = HostPacer(settings().scraper.host_gap_secs)
    failures = 0
    for key in keys:
        src = SOURCES.get(key)
        if src is None:
            typer.echo(f"unknown source: {key}. Known: {', '.join(SOURCES)}", err=True)
            raise typer.Exit(2)
        try:
            result = pipeline.run_source(src, force=force, pacer=pacer)
            typer.echo(
                f"{key}: {'indexed' if result.changed else 'unchanged'} ({result.chunks} chunks)"
            )
        except Exception as e:  # one bad source does not stop the rest
            failures += 1
            log.error("source failed", source=key, error=str(e))
            typer.echo(f"{key}: FAILED — {e}", err=True)
    if failures:
        raise typer.Exit(1)


def selected(source: str | None, *, all_sources: bool, category: str | None) -> list[str]:
    """The source keys one run covers: every source, one category, or the named source."""
    if sum((source is not None, all_sources, category is not None)) != 1:
        raise typer.BadParameter("give exactly one of a source key, --all, or --category")
    if all_sources:
        return list(SOURCES)
    if category is not None:
        keys = [key for key, src in SOURCES.items() if src.category == category]
        if not keys:
            known = ", ".join(sorted({src.category for src in SOURCES.values()}))
            raise typer.BadParameter(f"no source in category {category}. Known: {known}")
        return keys
    return [source] if source is not None else []


@app.command()
def serve() -> None:
    """Run the scraper: live queries, result indexing, scheduled sources, one job queue. Blocks.

    Requires the admin session; at a terminal it signs in first when the session is gone.
    """
    require_session()
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
    width = max(16, *(len(r["key"]) + 2 for r in rows))
    typer.echo(
        f"{'source':<{width}}{'category':<18}{'last attempt':<22}{'last change':<22}"
        f"{'versions':>9}{'chunks':>8}"
    )
    for r in rows:
        attempt = _stamp(r["last_attempt"])
        change = _stamp(r["last_fetch"])
        typer.echo(
            f"{r['key']:<{width}}{r['category']:<18}{attempt:<22}{change:<22}"
            f"{r['versions']:>9}{r['chunks']:>8}"
        )

    if queue:
        typer.echo(f"\n{'job kind':<16}{'status':<14}{'jobs':>8}")
        for q in queue:
            typer.echo(f"{q['kind']:<16}{q['status']:<14}{q['jobs']:>8}")


def _stamp(at: datetime | None) -> str:
    return at.strftime("%Y-%m-%d %H:%M UTC") if at else "-"


@app.command()
def login(
    if_needed: bool = typer.Option(
        False, "--if-needed", help="Check the saved session first; sign in only if it is gone."
    ),
    check: bool = typer.Option(
        False, "--check", help="Report whether the saved session works, then exit 0 or 1."
    ),
) -> None:
    """Capture the admin session the scraper requires (clubs, Sun Devil Central events).

    Opens the MyASU sign-in page, asks for your ASU username and password here, and waits for
    you to approve Duo. Only cookies are saved, never the password.
    """
    if check:
        if not _session_ok():
            typer.echo("admin session missing or expired", err=True)
            raise typer.Exit(1)
        typer.echo(f"admin session ok ({admin.state_path()})")
        return
    if if_needed:
        require_session()
        return
    _capture()


def _session_ok() -> bool:
    """admin.check, exiting 1 when the check page cannot be reached at all."""
    try:
        return admin.check()
    except FetchError as e:
        typer.echo(f"could not check the admin session: {e}", err=True)
        raise typer.Exit(1) from e


def require_session() -> None:
    """Exits unless the admin session works, signing in first when at a terminal."""
    if _session_ok():
        typer.echo(f"admin session ok ({admin.state_path()})")
        return
    typer.echo("admin session missing or expired; the scraper requires it", err=True)
    if not sys.stdin.isatty():
        typer.echo("not a terminal; run `just scraper login` first", err=True)
        raise typer.Exit(1)
    _capture()


def _capture() -> None:
    """Runs the interactive sign-in and saves the session. Exits 1 when it fails."""

    def ask() -> Credentials:
        username = typer.prompt("ASU username").strip()
        password = typer.prompt("ASU password", hide_input=True)
        return Credentials(username=username, password=password)

    typer.echo("Opening the MyASU sign-in page.")
    try:
        path = admin.capture_login(ask, typer.echo)
    except AuthError as e:
        typer.echo(f"login failed: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:  # a missing browser surfaces here
        typer.echo(f"login failed: {e}", err=True)
        typer.echo("If Chromium is missing, run `uv run playwright install chromium`.", err=True)
        raise typer.Exit(1) from e
    if path is None:
        raise typer.Exit(1)
    typer.echo(f"saved admin session to {path}")


@app.command()
def migrate() -> None:
    """Apply migrations/ to Postgres."""
    with postgres.connection() as conn:
        applied = postgres.migrate(conn)
    typer.echo("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
