"""`data generate|verify|redact|stats`."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Datasets: generate, verify, redact, stats."""


@app.command()
def stats() -> None:
    """Row counts per split."""
    raise NotImplementedError
