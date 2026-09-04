"""Settings from SPARKY_* env: postgres, object_store, embedding, scraper."""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Many ASU pages fill in content after load; give them a moment before extracting.
    wait_for_ms: int = 5_000
    only_main_content: bool = True


class Scraper(BaseModel):
    # Public ASU content is shared across every guild; the engine reads this tenant for all.
    tenant_id: str = "public"
    # "firecrawl" renders JS and returns markdown; "http" is the plain httpx + bs4 path.
    fetcher: str = "firecrawl"
    user_agent: str = "SparkyAI/2.0 (+https://github.com/ashworks1706/SparkyAI)"
    request_timeout_secs: float = 30.0
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 200
    parser_version: str = "bs4-text-v1"
    chunker_version: str = "para-1200-200-v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPARKY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres: Postgres = Postgres()
    object_store: ObjectStore = ObjectStore()
    embedding: Embedding = Embedding()
    firecrawl: Firecrawl = Firecrawl()
    scraper: Scraper = Scraper()


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings, loaded once."""
    return Settings()
