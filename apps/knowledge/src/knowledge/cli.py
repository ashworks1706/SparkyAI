"""`knowledge migrate`, `knowledge status`. Serving and scraping have their own entrypoints."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Knowledge service admin."""


@app.command()
def status() -> None:
    """Store health, document counts, last scrape per source."""
    raise NotImplementedError
