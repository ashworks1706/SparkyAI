"""`train sft --config configs/train/sft.yaml [--dry-run]`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from training.core.types import SftError
from training.posttrain import sft

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Post-training."""


@app.command("sft")
def sft_cmd(
    config: Path = typer.Option(Path("configs/train/sft.yaml")),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and data; no GPU."),
) -> None:
    """QLoRA SFT with Unsloth, exported to GGUF."""
    try:
        p = sft.plan(config)
        rprint(
            f"{p.base_model} · {p.examples} examples ({p.with_tool_calls} with tool calls) · "
            f"{p.epochs} epochs · seq {p.max_seq_length} → {p.output_dir}/gguf ({p.gguf_quant})"
        )
        if dry_run:
            return
        gguf = sft.train(config)
    except SftError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    rprint(f"[green]GGUF → {gguf}[/green]")
    rprint(f"serve it: SPARKY_CHAT_GGUF={gguf} just model")
