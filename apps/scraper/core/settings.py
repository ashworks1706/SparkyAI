"""Settings from sparky.toml and SPARKY_* env: postgres, object_store, embedding, summary,
scraper."""

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
    timeout_secs: float = 120.0


class Summary(BaseModel):
    """The chat endpoint a cluster summary is written by. Same server as embedding, chat model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: SecretStr = SecretStr("")
    name: str = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_secs: float = 120.0
    # Qwen3-style reasoning before the summary. On, it can spend max_tokens and leave no summary.
    thinking: bool = False


class Firecrawl(BaseModel):
    base_url: str = "http://localhost:3002"
    api_key: SecretStr = SecretStr("")
    timeout_ms: int = 60_000
    # Wait after page load before extracting.
    wait_for_ms: int = 5_000
    only_main_content: bool = True


class Search(BaseModel):
    """SearXNG, the metasearch engine behind the web search query source."""

    base_url: str = "http://localhost:8888"
    # Upstream engines asked on every search, comma-separated. SearXNG merges their results.
    engines: str = "google,brave,bing"
    language: str = "en-US"
    # 0 off, 1 moderate, 2 strict.
    safesearch: int = 1
    max_results: int = 8
    # Characters of each result snippet kept.
    snippet_chars: int = 300


class Telemetry(BaseModel):
    """OTLP/HTTP span export to PostHog and Phoenix. Each destination is independent: an empty
    host or project token turns PostHog off, an empty phoenix_url turns Phoenix off."""

    host: str = "http://localhost:8010"
    project_token: SecretStr = SecretStr("")
    traces_path: str = "/i/v1/traces"
    phoenix_url: str = ""
    export_timeout_secs: float = 10.0


class Scraper(BaseModel):
    # Public ASU content is shared across every guild; the engine reads this tenant for all.
    tenant_id: str = "public"
    # firecrawl renders JS and returns markdown; http is plain httpx + bs4 (Playwright when
    # the source needs JS).
    fetcher: Literal["firecrawl", "http"] = "firecrawl"
    user_agent: str = "SparkyAI/2.0 (+https://github.com/ashworks1706/SparkyAI)"
    request_timeout_secs: float = 30.0
    # Longest live query result handed back to the engine.
    query_max_chars: int = 6_000
    # Write each live query result into the retrieval index after its caller has the answer,
    # for query sources that allow it.
    index_live_results: bool = True
    # Longest a lane of scraper serve waits before looking at the queue again.
    serve_poll_secs: float = 0.5
    # How often scraper serve queues the scheduled runs that have fallen due.
    schedule_every_secs: float = 60.0
    # A background job running longer than this is taken to be left by a stopped process and
    # is queued again.
    job_lease_secs: float = 1800.0
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 200
    parser_version: str = "bs4-text-v1"
    # Quality floor: a run whose extracted text is below this fraction of the last indexed
    # version is refused.
    quality_floor_ratio: float = 0.5
    # The floor is skipped when the last version was shorter than this.
    quality_floor_min_chars: int = 500
    # The hierarchical index above the leaf chunks. Each cluster at each level costs one chat
    # call and one embedding call.
    tree_enabled: bool = False
    # Highest level built. Level 0 is the leaves, so 3 allows three levels of summary.
    tree_max_level: int = 3
    # How many rows of the level below one summary covers.
    tree_cluster_size: int = 5
    # A source with fewer leaves than this gets no tree.
    tree_min_chunks: int = 12

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
    summary: Summary = Summary()
    firecrawl: Firecrawl = Firecrawl()
    search: Search = Search()
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
