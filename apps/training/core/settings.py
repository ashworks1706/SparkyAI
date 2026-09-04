"""Settings from SPARKY_* env: where traces, Phoenix, the engine, and data live."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Training(BaseModel):
    # Engine under test and the trace directory it writes (relative to this app).
    engine_url: str = "http://localhost:8080"
    traces_dir: Path = Path("../../traces")
    phoenix_url: str = "http://localhost:6006"
    data_dir: Path = Path("data")
    cases_dir: Path = Path("evals/cases")
    baseline_path: Path = Path("evals/baseline.json")
    # Wall-clock budget per eval turn.
    request_timeout_secs: float = 180.0


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
