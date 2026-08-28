"""`sparky-ingest run <source>|--all`, `schedule`, `status`."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def status() -> None:
    """Last fetch time and version count per source."""
    raise NotImplementedError
