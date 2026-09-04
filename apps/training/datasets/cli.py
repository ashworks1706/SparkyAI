"""`data export`, `data verify`, `data stats`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint

from training.core.settings import settings
from training.core.types import TrainingExample
from training.datasets import export, redact, verify

app = typer.Typer(no_args_is_help=True)


def _read(path: Path) -> list[TrainingExample]:
    return [
        TrainingExample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _write(path: Path, examples: list[TrainingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(ex.model_dump_json() + "\n" for ex in examples))


@app.callback()
def main() -> None:
    """Datasets from real interactions."""


@app.command("export")
def export_cmd(
    out: Path = typer.Option(None, help="Defaults under .sparky/training/data/raw."),
) -> None:
    """Pull every complete `llm` span from Phoenix, redact it, and write raw JSONL."""
    out = out or settings().training.data_dir / "raw" / "llm_spans.jsonl"
    examples = [redact.redact_example(ex) for ex in export.export_examples()]
    _write(out, examples)
    rprint(f"[green]{len(examples)}[/green] examples → {out}")


@app.command("verify")
def verify_cmd(
    src: Path = typer.Option(None, help="Defaults under .sparky/training/data/raw."),
    out: Path = typer.Option(None, help="Defaults under .sparky/training/data/processed."),
) -> None:
    """Drop malformed and duplicate examples; write the training set."""
    src = src or settings().training.data_dir / "raw" / "llm_spans.jsonl"
    out = out or settings().training.data_dir / "processed" / "sft.jsonl"
    kept, reasons = verify.verify(_read(src))
    _write(out, kept)
    rprint(f"kept [green]{len(kept)}[/green] → {out}")
    for reason, n in sorted(reasons.items()):
        rprint(f"  dropped {n}: {reason}")


@app.command("stats")
def stats_cmd(src: Path = typer.Option(None)) -> None:
    """Counts by role, tool use, and session."""
    src = src or settings().training.data_dir / "processed" / "sft.jsonl"
    examples = _read(src)
    with_tools = sum(1 for e in examples if e.response.tool_calls)
    sessions = len({e.session_id for e in examples if e.session_id})
    turns = sum(len(e.messages) for e in examples)
    rprint(
        json.dumps(
            {
                "examples": len(examples),
                "with_tool_calls": with_tools,
                "sessions": sessions,
                "avg_messages": round(turns / len(examples), 1) if examples else 0,
            },
            indent=2,
        )
    )
