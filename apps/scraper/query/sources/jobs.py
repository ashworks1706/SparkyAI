"""On-campus student employment listings."""

from __future__ import annotations

from scraper.core.types import QuerySource
from scraper.sources.jobs import SOURCE as JOBS


def to_url(params: dict[str, str]) -> str:
    """The student employment page. It takes no parameters."""
    return JOBS.url


QUERY = QuerySource(
    key="jobs",
    description="Fetch ASU student employment listings and how to apply.",
    params=(),
    to_url=to_url,
    needs_js=JOBS.needs_js,
    extractor=JOBS.extractor,
)
