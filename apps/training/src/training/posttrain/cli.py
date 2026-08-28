"""`train sft|dpo|grpo|merge --config configs/train/<name>.yaml`."""

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Post-training: SFT → DPO → GRPO."""


CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "train"


@app.command()
def configs() -> None:
    """List available training configs."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        typer.echo(path.stem)
