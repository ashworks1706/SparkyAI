"""Bookable study room slots at an ASU library on one date."""

from __future__ import annotations

import datetime

from scraper.core.types import QueryError, QueryParam, QuerySource
from scraper.ingest.extract import form_page_text
from scraper.query.params import choices, code, text, url

# LibCal location and group ids per library.
_LIBRARIES = {
    "hayden": "13858:28619",
    "noble": "1702:2897",
    "fletcher": "1703:2898",
    "west": "1707:28611",
    "polytechnic": "1704:2899",
}


def to_url(params: dict[str, str]) -> str:
    """The LibCal availability page for one library and date."""
    ids = code(params, "library", _LIBRARIES)
    date = text(params, "date")
    try:
        datetime.date.fromisoformat(date)
    except ValueError as e:
        raise QueryError(f"date must look like 2026-09-14, got {date!r}") from e
    lid, gid = ids.split(":")
    return url(
        "https://asu.libcal.com/r/accessible/availability",
        [
            ("lid", lid),
            ("gid", gid),
            ("zone", "0"),
            ("space", "0"),
            ("capacity", "2"),
            ("date", date),
        ],
    )


QUERY = QuerySource(
    key="study_rooms",
    description="List open study room slots at one ASU library on one date.",
    params=(
        QueryParam("library", "Library.", required=True, choices=choices(_LIBRARIES)),
        QueryParam("date", "Date to check.", required=True, example="2026-09-14"),
    ),
    to_url=to_url,
    needs_js=True,
    extractor=form_page_text,
)
