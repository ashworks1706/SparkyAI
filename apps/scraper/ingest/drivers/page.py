"""One page load in a Playwright browser context, turned into a Fetched page."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

from playwright.sync_api import BrowserContext, Route

from scraper.core.settings import settings
from scraper.core.types import Fetched, FetchError, FetchRejected


@dataclass(frozen=True)
class Loaded:
    """What one navigation produced: the HTTP status, where it ended, and the rendered DOM."""

    status: int
    url: str
    html: str


@lru_cache(maxsize=1)
def _browsers() -> threading.BoundedSemaphore:
    """One permit per Chromium allowed open at once, shared by every lane of the process."""
    return threading.BoundedSemaphore(max(1, settings().scraper.max_browsers))


@contextmanager
def browser_slot() -> Iterator[None]:
    """Holds one of scraper.max_browsers permits for as long as a browser is open."""
    with _browsers():
        yield


def skip_heavy(context: BrowserContext) -> None:
    """Aborts every request whose resource type is listed in scraper.browser_skip."""
    skipped = {t.strip() for t in settings().scraper.browser_skip.split(",") if t.strip()}
    if not skipped:
        return

    def route(r: Route) -> None:
        if r.request.resource_type in skipped:
            r.abort()
        else:
            r.continue_()

    context.route("**/*", route)


def load(context: BrowserContext, url: str, timeout_secs: float) -> Loaded:
    """Navigates a new page to url, waits for the network to settle, and reads the DOM."""
    page = context.new_page()
    try:
        response = page.goto(url, wait_until="networkidle", timeout=int(timeout_secs * 1000))
        return Loaded(status=response.status if response else 0, url=page.url, html=page.content())
    finally:
        page.close()


def to_fetched(requested: str, loaded: Loaded) -> Fetched:
    """The loaded page as Fetched. No response is a fetch error; 4xx is rejected, 5xx retried."""
    if loaded.status == 0:
        raise FetchError(f"{requested}: navigation returned no response")
    if 400 <= loaded.status < 500:
        raise FetchRejected(f"{requested}: {loaded.status}")
    if loaded.status >= 500:
        raise FetchError(f"{requested}: {loaded.status}")
    return Fetched(
        url=loaded.url,
        status=loaded.status,
        body=loaded.html.encode("utf-8"),
        content_type="text/html",
    )
