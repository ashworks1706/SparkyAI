"""ASU News stories, searched by keyword."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import text, url


def to_url(params: dict[str, str]) -> str:
    """The news search results for a keyword."""
    return url("https://news.asu.edu/search", [("search", text(params, "keywords"))])


QUERY = QuerySource(
    key="news",
    description="Search ASU News for recent stories.",
    params=(QueryParam("keywords", "What the story is about.", required=True, example="robotics"),),
    to_url=to_url,
)
