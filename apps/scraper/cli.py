"""`scraper run <source>|--all`, `scraper schedule`, `scraper status`, `scraper migrate`."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Scraper: keep the retrieval index fresh."""


@app.command()
def run(source: str | None = None, all_sources: bool = False) -> None:
    """Fetch, extract, chunk, embed, and index one source or every enabled source."""
    raise NotImplementedError


@app.command()
def schedule() -> None:
    """Run enabled sources on their `fetch_every` interval."""
    raise NotImplementedError


@app.command()
def status() -> None:
    """Last fetch time and version count per source."""
    raise NotImplementedError


@app.command()
def migrate() -> None:
    """Apply `migrations/` to Postgres."""
    raise NotImplementedError
