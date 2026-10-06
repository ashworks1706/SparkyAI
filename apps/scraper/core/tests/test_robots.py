"""robots.txt: what the scraper may fetch, and what happens when a site says no."""

from __future__ import annotations

import pytest
from scraper.core.types import FetchError, FetchRejected
from scraper.ingest import fetch, robots

AGENT = "SparkyAI/2.0 (+https://github.com/ashworks1706/SparkyAI)"

RULES = """
User-agent: *
Disallow: /private/

User-agent: SparkyAI
Disallow: /search
"""


def test_a_disallowed_path_is_refused_and_the_rest_is_allowed() -> None:
    rules = robots.rules(200, RULES)
    assert rules.can_fetch(AGENT, "https://lib.asu.edu/hours")
    assert not rules.can_fetch(AGENT, "https://lib.asu.edu/search?q=x"), "the named agent group"
    assert rules.can_fetch(AGENT, "https://lib.asu.edu/private/x"), "a named group replaces *"
    assert not rules.can_fetch("OtherBot/1.0", "https://lib.asu.edu/private/x")


def test_a_missing_robots_file_allows_everything() -> None:
    assert robots.rules(404, "").can_fetch(AGENT, "https://lib.asu.edu/hours")


def test_a_site_that_cannot_serve_robots_is_not_fetched_now() -> None:
    with pytest.raises(FetchError):
        robots.rules(503, "")


def test_a_page_robots_disallows_is_rejected_before_any_fetch(monkeypatch) -> None:
    monkeypatch.setattr(robots, "allowed", lambda _url: False)
    monkeypatch.setattr(fetch, "fetch_http", lambda _url: pytest.fail("fetched a disallowed page"))
    monkeypatch.setattr(
        fetch, "fetch_firecrawl", lambda _url: pytest.fail("fetched a disallowed page")
    )
    with pytest.raises(FetchRejected, match="robots.txt"):
        fetch.fetch("https://lib.asu.edu/search?q=x")
