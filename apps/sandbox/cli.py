"""`sandbox serve`, `sandbox sessions`, `sandbox cleanup`."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Sandbox: isolated browser worker."""


@app.command()
def sessions() -> None:
    """List live sessions and their expiry."""
    raise NotImplementedError
