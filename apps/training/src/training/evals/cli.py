"""`eval run <suite> --model <name>`, `eval bfcl`, `eval lm-eval`."""

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Evals: Inspect suites, BFCL, lm-eval."""


SUITES = Path(__file__).resolve().parent / "suites"


@app.command()
def suites() -> None:
    """List available eval suites."""
    for path in sorted(SUITES.glob("*.py")):
        if path.stem != "__init__":
            typer.echo(path.stem)
