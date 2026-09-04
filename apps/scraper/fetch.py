"""httpx fetch; Playwright fallback for JS-rendered pages."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scraper.settings import settings


@dataclass(frozen=True)
class Fetched:
    url: str
    status: int
    body: bytes
    content_type: str


class FetchError(RuntimeError):
    """Transient failure; retried."""


class FetchRejected(RuntimeError):
    """4xx from the origin; not retried."""


@retry(
    retry=retry_if_exception_type((httpx.TransportError, FetchError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    reraise=True,
)
def fetch_http(url: str) -> Fetched:
    """One GET with retries on transport errors and 5xx."""
    s = settings().scraper
    with httpx.Client(
        headers={"User-Agent": s.user_agent},
        timeout=s.request_timeout_secs,
        follow_redirects=True,
    ) as http:
        r = http.get(url)
    if r.status_code >= 500:
        raise FetchError(f"{url} returned {r.status_code}")
    if r.status_code >= 400:
        raise FetchRejected(f"{url} returned {r.status_code}")
    return Fetched(
        url=str(r.url),
        status=r.status_code,
        body=r.content,
        content_type=r.headers.get("content-type", "text/html"),
    )


def fetch_rendered(url: str) -> Fetched:
    """Loads the page in headless Chromium and returns the rendered DOM."""
    from playwright.sync_api import sync_playwright

    s = settings().scraper
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=s.user_agent)
            page.goto(url, wait_until="networkidle", timeout=int(s.request_timeout_secs * 1000))
            html = page.content()
            final_url = page.url
        finally:
            browser.close()
    return Fetched(url=final_url, status=200, body=html.encode("utf-8"), content_type="text/html")


def fetch(url: str, *, needs_js: bool = False) -> Fetched:
    return fetch_rendered(url) if needs_js else fetch_http(url)
