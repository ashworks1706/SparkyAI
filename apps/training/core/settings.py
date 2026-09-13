"""Settings from sparky.toml and SPARKY_* env: where traces, PostHog, the engine, and data live."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _toml_files() -> tuple[Path, ...]:
    """Where sparky.toml is looked for. SPARKY_CONFIG_FILE overrides both paths."""
    override = os.environ.get("SPARKY_CONFIG_FILE")
    if override:
        return (Path(override),)
    return (Path("../../sparky.toml"), Path("sparky.toml"))


class Training(BaseModel):
    engine_url: str = "http://localhost:8080"
    # Bearer token the engine /chat route requires.
    engine_service_token: SecretStr = SecretStr("")
    posthog_host: str = "http://localhost:8010"
    posthog_project_id: str = ""
    # Personal API key with the Query Read scope. A project token is rejected.
    posthog_api_key: SecretStr = SecretStr("")
    posthog_page_rows: int = 5000
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment first, then .env, then sparky.toml. Earlier sources win."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=_toml_files()),
        )


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
