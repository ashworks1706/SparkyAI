"""
`knowledge-scraper run <source>|--all`, `knowledge-scraper schedule`, `knowledge-scraper
status`.
"""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Scraper: keep the knowledge index fresh."""


@app.command()
def status() -> None:
    """Last fetch time and version count per source."""
    raise NotImplementedError
