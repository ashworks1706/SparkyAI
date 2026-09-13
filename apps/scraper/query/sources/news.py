"""ASU News stories: the newest on the front page, or a search by keyword."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import text, url

# Words that ask for recent stories rather than name a topic.
_NOT_A_TOPIC = {"latest", "recent", "new", "newest", "today", "news", "asu", "stories", "top"}


def to_url(params: dict[str, str]) -> str:
    """The news search results for a topic, or the front page without one."""
    keywords = text(params, "keywords")
    if not set(keywords.lower().split()) - _NOT_A_TOPIC:
        return "https://news.asu.edu/"
    return url("https://news.asu.edu/search", [("search", keywords)])


QUERY = QuerySource(
    key="news",
    description="The newest ASU News stories, or a search of ASU News by topic.",
    params=(QueryParam("keywords", "Topic to search for.", example="robotics"),),
    to_url=to_url,
    category="news",
)
