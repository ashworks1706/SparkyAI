from pathlib import Path

from training.core.settings import Training


def test_generated_files_share_one_state_directory() -> None:
    cfg = Training(state_dir=Path("state"))

    assert cfg.traces_dir == Path("state/traces")
    assert cfg.eval_report_path == Path("state/training/evals/last.json")
