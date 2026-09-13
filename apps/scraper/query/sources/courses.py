"""The live ASU class search: sections, meeting days, and open seats for one term."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import choices, code, codes, term_code, text, url

_DAYS = {
    "monday": "MON",
    "tuesday": "TUES",
    "wednesday": "WED",
    "thursday": "THURS",
    "friday": "FRI",
    "saturday": "SAT",
    "sunday": "SUN",
}

_LEVELS = {
    "lower division": "lowerdivision",
    "upper division": "upperdivision",
    "undergraduate": "undergrad",
    "graduate": "grad",
    "100-199": "100-199",
    "200-299": "200-299",
    "300-399": "300-399",
    "400-499": "400-499",
}

_SESSIONS = {"a": "A", "b": "B", "c": "C", "other": "DYN"}


def to_url(params: dict[str, str]) -> str:
    """The class list URL. Only term is required; everything else narrows."""
    return url(
        "https://catalog.apps.asu.edu/catalog/classes/classlist",
        [
            ("advanced", "true"),
            ("campusOrOnlineSelection", "A"),
            ("honors", "F"),
            ("promod", "F"),
            ("term", term_code(text(params, "term"))),
            ("keywords", text(params, "keywords")),
            ("level", ",".join(codes(params, "level", _LEVELS))),
            ("daysOfWeek", ",".join(codes(params, "days", _DAYS))),
            ("session", code(params, "session", _SESSIONS)),
            ("searchType", "open" if text(params, "open_only").lower() == "true" else "all"),
        ],
    )


QUERY = QuerySource(
    key="courses",
    description=(
        "Search the ASU class catalog for one term: sections, instructors, meeting days, "
        "and open seats."
    ),
    params=(
        QueryParam("term", "Term to search.", required=True, example="Fall 2026"),
        QueryParam("keywords", "Subject, course number or title.", example="CSE 310"),
        QueryParam("level", "Course levels.", choices=choices(_LEVELS), many=True),
        QueryParam("days", "Meeting days.", choices=choices(_DAYS), many=True),
        QueryParam("session", "Session within the term.", choices=choices(_SESSIONS)),
        QueryParam("open_only", "Only sections with open seats.", choices=("true", "false")),
    ),
    to_url=to_url,
    needs_js=True,
)
