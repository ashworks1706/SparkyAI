"""ASU Library hours for this week, every location."""

from __future__ import annotations

from scraper.core.types import QuerySource
from scraper.sources.library_hours import SOURCE as HOURS


def to_url(params: dict[str, str]) -> str:
    """The hours page. It takes no parameters."""
    return HOURS.url


QUERY = QuerySource(
    key="library_hours",
    description="Fetch this week's opening hours for every ASU library.",
    params=(),
    to_url=to_url,
    needs_js=HOURS.needs_js,
    extractor=HOURS.extractor,
)
