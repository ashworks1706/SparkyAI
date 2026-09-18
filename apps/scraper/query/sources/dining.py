"""Dining hours for one campus, read from the Sun Devil Hospitality hours PDF."""

from __future__ import annotations

from scraper.core.types import QueryError, QueryParam, QuerySource
from scraper.query.params import text
from scraper.sources.dining_hours import SOURCES, extract_dining_hours

_BY_CAMPUS = {source.key.removeprefix("dining_hours_"): source for source in SOURCES}

_CAMPUSES = tuple(_BY_CAMPUS)


def to_url(params: dict[str, str]) -> str:
    """The hours PDF of the campus named."""
    campus = text(params, "campus").lower()
    source = _BY_CAMPUS.get(campus)
    if source is None:
        raise QueryError(f"campus {campus!r} is not one of: {', '.join(_CAMPUSES)}")
    return source.url


QUERY = QuerySource(
    key="dining",
    description=(
        "Fetch dining hall and campus restaurant hours for one ASU campus, one line per venue."
    ),
    params=(
        QueryParam(
            "campus",
            "Campus whose dining hours to read.",
            required=True,
            choices=_CAMPUSES,
            example="tempe",
        ),
    ),
    to_url=to_url,
    extractor=extract_dining_hours,
    category="dining",
)
