"""The ASU events calendar, searched by keyword."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import text, url


def to_url(params: dict[str, str]) -> str:
    """The events listing, narrowed to a keyword when one was given."""
    return url("https://asuevents.asu.edu/home", [("searchText", text(params, "keywords"))])


QUERY = QuerySource(
    key="events",
    description="Search the ASU events calendar for upcoming events.",
    params=(QueryParam("keywords", "What the event is about.", example="career fair"),),
    to_url=to_url,
    needs_js=True,
)
