"""Settings from sparky.toml and SPARKY_* env: postgres, object_store, embedding, scraper."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

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


class Postgres(BaseModel):
    url: SecretStr = SecretStr("postgres://sparky:sparky@localhost:5432/sparky")


class ObjectStore(BaseModel):
    endpoint: str = "http://localhost:9000"
    bucket: str = "sparky-snapshots"
    access_key: str = "minioadmin"
    secret_key: SecretStr = SecretStr("minioadmin")
    region: str = "us-east-1"


class Embedding(BaseModel):
    base_url: str = "http://localhost:8001/v1"
    api_key: SecretStr = SecretStr("")
    name: str = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
    dim: int = 1024
    batch_size: int = 32


class Firecrawl(BaseModel):
    base_url: str = "http://localhost:3002"
    api_key: SecretStr = SecretStr("")
    timeout_ms: int = 60_000
    # ASU pages fill in content after load; wait before extracting.
    wait_for_ms: int = 5_000
    only_main_content: bool = True


class Telemetry(BaseModel):
    # Phoenix locally; empty disables export.
    otlp_endpoint: str = "http://localhost:4317"


class Scraper(BaseModel):
    # Public ASU content is shared across every guild; the engine reads this tenant for all.
    tenant_id: str = "public"
    # firecrawl renders JS and returns markdown; http is plain httpx + bs4 (Playwright when
    # the source needs JS).
    fetcher: Literal["firecrawl", "http"] = "firecrawl"
    user_agent: str = "SparkyAI/2.0 (+https://github.com/ashworks1706/SparkyAI)"
    request_timeout_secs: float = 30.0
    # Longest live query result handed back to the engine.
    query_max_chars: int = 12_000
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 200
    parser_version: str = "bs4-text-v1"
    # Quality floor: a run whose extracted text is below this fraction of the last indexed
    # version is refused rather than written over the index.
    quality_floor_ratio: float = 0.5
    # The floor is skipped when the last version was shorter than this, where a swing of a few
    # hundred characters is ordinary.
    quality_floor_min_chars: int = 500

    def chunker_version(self) -> str:
        """Records the settings the chunks were cut with."""
        return f"para-{self.chunk_chars}-{self.chunk_overlap_chars}-v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPARKY_",
        env_nested_delimiter="__",
        env_file="../../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres: Postgres = Postgres()
    object_store: ObjectStore = ObjectStore()
    embedding: Embedding = Embedding()
    firecrawl: Firecrawl = Firecrawl()
    telemetry: Telemetry = Telemetry()
    scraper: Scraper = Scraper()

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
    """Process-wide settings, loaded once."""
    return Settings()
