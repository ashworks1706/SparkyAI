from pathlib import Path

from training.core.settings import Training


def test_generated_files_share_one_state_directory() -> None:
    cfg = Training(state_dir=Path("state"))

    assert cfg.traces_dir == Path("state/traces")
    assert cfg.data_dir == Path("state/training/data")
    assert cfg.eval_report_path == Path("state/training/evals/last.json")
    assert cfg.output_dir == Path("state/training/outputs")
