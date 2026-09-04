"""Settings from SPARKY_* env: where traces, Phoenix, the engine, and data live."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Training(BaseModel):
    engine_url: str = "http://localhost:8080"
    phoenix_url: str = "http://localhost:6006"
    state_dir: Path = Path("../../.sparky")
    cases_dir: Path = Path("evals/cases")
    baseline_path: Path = Path("evals/baseline.json")
    request_timeout_secs: float = 180.0

    @property
    def traces_dir(self) -> Path:
        return self.state_dir / "traces"

    @property
    def data_dir(self) -> Path:
        return self.state_dir / "training" / "data"

    @property
    def eval_report_path(self) -> Path:
        return self.state_dir / "training" / "evals" / "last.json"

    @property
    def output_dir(self) -> Path:
        return self.state_dir / "training" / "outputs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPARKY_",
        env_nested_delimiter="__",
        env_file="../../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    training: Training = Training()


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
