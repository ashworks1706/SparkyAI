"""httpx fetch; Playwright fallback for JS-rendered pages."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scraper.core.settings import settings
from scraper.core.types import Fetched, FetchError, FetchRejected


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


def fetch_firecrawl(url: str) -> Fetched:
    """Scrapes through self-hosted Firecrawl: JS rendered, boilerplate stripped, markdown out."""
    cfg = settings().firecrawl
    headers = {}
    key = cfg.api_key.get_secret_value()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(
        base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=cfg.timeout_ms / 1000 + 10
    ) as http:
        r = http.post(
            "/v2/scrape",
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": cfg.only_main_content,
                "waitFor": cfg.wait_for_ms,
                "timeout": cfg.timeout_ms,
            },
        )
    if r.status_code >= 500:
        raise FetchError(f"firecrawl returned {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        raise FetchRejected(f"firecrawl returned {r.status_code}: {r.text[:200]}")
    payload = r.json()
    if not payload.get("success"):
        raise FetchError(f"firecrawl failed: {str(payload.get('error'))[:200]}")
    data = payload.get("data") or {}
    meta = data.get("metadata") or {}
    markdown = data.get("markdown") or ""
    status = int(meta.get("statusCode") or 200)
    if status >= 400:
        raise FetchRejected(f"{url} returned {status}")
    if not markdown.strip():
        raise FetchError(f"firecrawl returned no content for {url}")
    return Fetched(
        url=str(meta.get("url") or meta.get("sourceURL") or url),
        status=status,
        body=markdown.encode("utf-8"),
        content_type="text/markdown",
        title=meta.get("title"),
        text=markdown,
    )


def fetch(url: str, *, needs_js: bool = False) -> Fetched:
    """Fetches by the configured fetcher. Firecrawl renders JS itself, so `needs_js` only
    matters on the plain HTTP path."""
    if settings().scraper.fetcher == "firecrawl":
        return fetch_firecrawl(url)
    return fetch_rendered(url) if needs_js else fetch_http(url)
