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
    # Pooled connections the scraper holds at most; covers every serve lane plus the scheduler.
    scraper_pool_max: int = 8


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


class Auth(BaseModel):
    """Admin authenticated browser session for login-gated ASU sources. Not per-user MyASU.

    Only the browser storage state is saved, never a password.
    """

    # Where the captured storage state is read from and written to.
    storage_state_path: str = "../../.sparky/auth/admin_state.json"
    # The MyASU page scraper login opens first; it redirects to the ASU sign-in form.
    login_url: str = "https://my.asu.edu/"
    # The login-gated service entered after MyASU, and the link on it that starts single sign-on.
    service_login_url: str = "https://sundevilcentral.eoss.asu.edu/webapp/auth/login"
    service_sso_text: str = "SSO Login"
    # A page that only loads when signed in; scraper login --if-needed checks the session with it.
    check_url: str = "https://sundevilcentral.eoss.asu.edu/events"
    # Hosts a sign-in passes through, comma-separated. A page on one has not finished signing in.
    sso_hosts: str = "weblogin.asu.edu,duosecurity.com,campusgroups.com"
    # Hosts that mean the browser is not signed in; a fetch redirected to one is an expired session.
    login_hosts: str = "weblogin.asu.edu,cas.asu.edu,login.microsoftonline.com,idp.asu.edu"
    # Paths of a service's own sign-in page, comma-separated.
    login_paths: str = "/webapp/auth/login"
    # Run the sign-in browser without a window. Credentials and Duo prompts go through the console.
    login_headless: bool = True
    # Password attempts before scraper login gives up.
    login_attempts: int = 3
    # Longest scraper login waits for Duo to be approved.
    duo_timeout_secs: float = 120.0
    # Longest a headless authenticated fetch waits for a page to settle.
    nav_timeout_secs: float = 60.0
    # Tries an authenticated fetch gets when the browser hits a network error.
    fetch_attempts: int = 3


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
    """OTLP/HTTP span export to Phoenix. An empty phoenix_url turns export off."""

    phoenix_url: str = ""
    phoenix_api_key: SecretStr = SecretStr("")
    project_name: str = "sparky"
    export_timeout_secs: float = 10.0


class Scraper(BaseModel):
    # Tenant public ASU content is stored under; the engine reads it for every guild.
    tenant_id: str = "public"
    # firecrawl renders JS to markdown; http is httpx + bs4, with Playwright for JS sources.
    fetcher: Literal["firecrawl", "http"] = "firecrawl"
    user_agent: str = "SparkyAI/2.0 (+https://github.com/ashworks1706/SparkyAI)"
    request_timeout_secs: float = 30.0
    # Longest live query result handed back to the engine.
    query_max_chars: int = 30_000
    # Index live query results after the caller has its answer, where the source allows it.
    index_live_results: bool = True
    # Longest a lane of scraper serve waits before looking at the queue again.
    serve_poll_secs: float = 0.5
    # Live lanes scraper serve runs, each answering one live query at a time.
    live_workers: int = 4
    # Playwright resource types the browser drivers never load, comma-separated.
    browser_skip: str = "image,media,font"
    # Chromium browsers open at once across every lane. A fetch past it waits for one to close.
    max_browsers: int = 2
    # Largest page body a plain HTTP fetch reads, in bytes. A larger page is refused.
    max_page_bytes: int = 10_000_000
    # Versions kept per source, newest first, with their snapshots. 0 keeps every version.
    keep_versions: int = 5
    # How often scraper serve queues the scheduled runs that have fallen due.
    schedule_every_secs: float = 60.0
    # A job running longer than this is treated as abandoned and requeued.
    job_lease_secs: float = 1800.0
    # Shortest gap between two scheduled fetches to the same host. Zero fetches back to back.
    host_gap_secs: float = 5.0
    # Queued live_index jobs past which a live result is answered but not indexed.
    index_backlog_limit: int = 500
    # How long a finished job is kept before it is removed. Zero keeps every job forever.
    job_retention_hours: float = 72.0
    # Finished jobs removed per scheduling cycle.
    job_prune_batch: int = 5000
    # Characters per leaf chunk; retrieval.window reads the neighbours back with a hit.
    chunk_chars: int = 300
    chunk_overlap_chars: int = 0
    parser_version: str = "bs4-text-v1"
    # A run whose text is below this fraction of the last indexed version is refused.
    quality_floor_ratio: float = 0.5
    # The floor is skipped when the last version was shorter than this.
    quality_floor_min_chars: int = 500
    # Summary tree over leaf chunks; each cluster costs one chat call and one embedding call.
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
    auth: Auth = Auth()
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
