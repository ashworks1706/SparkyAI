from pathlib import Path

from evals.core.settings import Evals


def test_generated_files_share_one_state_directory() -> None:
    cfg = Evals(state_dir=Path("state"))

    assert cfg.traces_dir == Path("state/traces")
    assert cfg.eval_report_path == Path("state/evals/last.json")
