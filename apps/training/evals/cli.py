"""`eval run`, `eval compare`, `eval baseline`."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from training.core.settings import settings
from training.core.types import CaseResult, EvalReport, SuiteReport
from training.evals import runner

app = typer.Typer(no_args_is_help=True)

SUITES = [
    "tool_selection",
    "tool_args",
    "grounding",
    "memory",
    "permissions",
    "clarification",
    "refusal",
    "latency",
]


def _suite(name: str):
    if name not in SUITES:
        raise typer.BadParameter(f"unknown suite {name}; known: {', '.join(SUITES)}")
    return importlib.import_module(f"training.evals.suites.{name}")


def _report(results: list[CaseResult], engine_url: str, case_count: int) -> EvalReport:
    suites = []
    for name in SUITES:
        rs = [r for r in results if r.suite == name]
        if rs:
            suites.append(
                SuiteReport(suite=name, passed=sum(r.score.passed for r in rs), total=len(rs))
            )
    return EvalReport(engine_url=engine_url, cases=case_count, results=results, suites=suites)


def _print(report: EvalReport) -> None:
    table = Table(title=f"evals against {report.engine_url}")
    table.add_column("suite")
    table.add_column("passed", justify="right")
    table.add_column("rate", justify="right")
    for s in report.suites:
        table.add_row(s.suite, f"{s.passed}/{s.total}", f"{s.rate:.0%}")
    rprint(table)
    for r in report.results:
        if not r.score.passed:
            rprint(f"  [red]FAIL[/red] {r.suite} · {r.case_id}: {r.score.detail}")


@app.callback()
def main() -> None:
    """Engine evals with a baseline gate."""


@app.command("run")
def run_cmd(
    suite: list[str] = typer.Option(None, "--suite", help="Suites to run; default all."),
    out: Path = typer.Option(Path("evals/last.json"), help="Where to write the report."),
    engine_url: str = typer.Option(None, help="Defaults to SPARKY_TRAINING__ENGINE_URL."),
) -> None:
    """Run the golden cases against a live engine and score them."""
    wanted = set(suite or SUITES)
    cases = [c for c in runner.load_cases() if wanted & set(c.suites)]
    results: list[CaseResult] = []
    for case in cases:
        try:
            turns = runner.run_case(case, engine_url)
        except runner.RunnerError as e:
            raise typer.Exit(f"case {case.id}: {e}") from e
        for name in case.suites:
            if name not in wanted:
                continue
            score = _suite(name).score(case, turns)
            results.append(
                CaseResult(
                    case_id=case.id, suite=name, score=score, request_id=turns[-1].request_id
                )
            )
    report = _report(results, engine_url or settings().training.engine_url, len(cases))
    out.write_text(report.model_dump_json(indent=2))
    _print(report)
    rprint(f"report → {out}")


@app.command("baseline")
def baseline_cmd(src: Path = typer.Option(Path("evals/last.json"))) -> None:
    """Promote the last report's suite rates to the committed baseline."""
    report = EvalReport.model_validate_json(src.read_text())
    baseline = {s.suite: {"passed": s.passed, "total": s.total} for s in report.suites}
    path = settings().training.baseline_path
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    rprint(f"baseline → {path}")


@app.command("compare")
def compare_cmd(
    src: Path = typer.Option(Path("evals/last.json")),
    tolerance: float = typer.Option(0.0, help="Allowed drop in pass rate per suite."),
) -> None:
    """Fail when any suite's pass rate fell below the baseline."""
    path = settings().training.baseline_path
    if not path.exists():
        raise typer.Exit(f"no baseline at {path}; run `eval run` then `eval baseline` first")
    baseline = json.loads(path.read_text())
    report = EvalReport.model_validate_json(src.read_text())
    regressions = []
    for s in report.suites:
        b = baseline.get(s.suite)
        if not b or not b["total"]:
            continue
        before = b["passed"] / b["total"]
        if s.rate + tolerance < before:
            regressions.append(f"{s.suite}: {before:.0%} → {s.rate:.0%}")
    if regressions:
        rprint("[red]regressions:[/red] " + "; ".join(regressions))
        raise typer.Exit(1)
    rprint("[green]no regressions against baseline[/green]")
