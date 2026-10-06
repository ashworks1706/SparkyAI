"""robots.txt of each site, read with the scraper's user agent and kept for robots_cache_secs."""

from __future__ import annotations

import time
from functools import lru_cache
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from scraper.core.settings import settings
from scraper.core.types import FetchError


def allowed(url: str) -> bool:
    """Whether the robots.txt of url's site lets the scraper's user agent fetch url."""
    s = settings().scraper
    parts = urlsplit(url)
    window = int(time.time() // s.robots_cache_secs)
    return _site(f"{parts.scheme}://{parts.netloc}", window).can_fetch(s.user_agent, url)


@lru_cache(maxsize=256)
def _site(origin: str, window: int) -> RobotFileParser:
    """The parsed robots.txt of origin. A new window expires the cached entry."""
    s = settings().scraper
    try:
        r = httpx.get(
            f"{origin}/robots.txt",
            headers={"User-Agent": s.user_agent},
            timeout=s.request_timeout_secs,
            follow_redirects=True,
        )
    except httpx.TransportError as e:
        raise FetchError(f"robots.txt of {origin} could not be read: {e}") from e
    return rules(r.status_code, r.text)


def rules(status: int, body: str) -> RobotFileParser:
    """Rules from one robots.txt response. 4xx allows everything; 5xx raises FetchError."""
    if status >= 500:
        raise FetchError(f"robots.txt returned {status}; the site is not fetched until it answers")
    parser = RobotFileParser()
    if status >= 400:
        parser.allow_all = True
    else:
        parser.parse(body.splitlines())
    return parser
