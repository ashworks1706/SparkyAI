"""The registry of live parameterized sources, and the URL each one builds.

Distinct from sources/, which are pages fetched on a schedule into the retrieval index. A query
source covers a combinatorial space too large to enumerate, like every term by subject by level
of the class catalog.
"""

from __future__ import annotations

import urllib.parse

from scraper.core.types import QueryError, QueryParam, QuerySource

# ASU term codes are 2 plus the two-digit calendar year plus the session digit.
# Fall 2026 is 2267.
_TERM_DIGIT = {"spring": "1", "summer": "4", "fall": "7"}

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


def term_code(term: str) -> str:
    """Fall 2026 to 2267. Raises QueryError on anything else."""
    parts = term.strip().split()
    if len(parts) != 2:
        raise QueryError(f"term must look like 'Fall 2026', got {term!r}")
    season, year = parts[0].lower(), parts[1]
    if season not in _TERM_DIGIT:
        raise QueryError(f"term season must be spring, summer or fall, got {parts[0]!r}")
    if not (year.isdigit() and len(year) == 4):
        raise QueryError(f"term year must be four digits, got {year!r}")
    return f"2{year[2:]}{_TERM_DIGIT[season]}"


def _mapped(value: str | None, mapping: dict[str, str], what: str) -> str:
    """Comma-joined codes for a comma-separated list of human names."""
    if not value:
        return ""
    codes = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise QueryError(f"{what} {item.strip()!r} is not one of: {', '.join(sorted(mapping))}")
        codes.append(mapping[key])
    return ",".join(codes)


def _classlist_url(params: dict[str, str]) -> str:
    """ASU class catalog search. Only term is required; everything else narrows."""
    query = {
        "advanced": "true",
        "campusOrOnlineSelection": "A",
        "honors": "F",
        "promod": "F",
        "term": term_code(params["term"]),
        "keywords": params.get("keywords", "").strip(),
        "level": _mapped(params.get("level"), _LEVELS, "level"),
        "daysOfWeek": _mapped(params.get("days"), _DAYS, "day"),
        "searchType": "open" if params.get("open_only", "").lower() == "true" else "all",
    }
    kept = {k: v for k, v in query.items() if v}
    return "https://catalog.apps.asu.edu/catalog/classes/classlist?" + urllib.parse.urlencode(kept)


def _scholarship_url(params: dict[str, str]) -> str:
    """ASU Global Education scholarship search."""
    query = {
        k: v
        for k, v in {
            "academiclevel": params.get("academic_level", "").strip(),
            "citizenship_status": params.get("citizenship", "").strip(),
            "gpa": params.get("gpa", "").strip(),
        }.items()
        if v
    }
    base = "https://goglobal.asu.edu/scholarship-search"
    return f"{base}?{urllib.parse.urlencode(query)}" if query else base


QUERY_SOURCES: dict[str, QuerySource] = {
    q.key: q
    for q in (
        QuerySource(
            key="class_search",
            description=(
                "Search the live ASU class catalog for a specific term. Use for course "
                "availability, meeting days and open seats, which the knowledge base does not "
                "carry."
            ),
            params=(
                QueryParam("term", "Term to search, e.g. 'Fall 2026'.", required=True),
                QueryParam("keywords", "Subject, course number or title, e.g. 'CSE 310'."),
                QueryParam(
                    "level",
                    "Comma-separated: undergraduate, graduate, lower division, upper "
                    "division, 100-199, 200-299, 300-399, 400-499.",
                    example="undergraduate",
                ),
                QueryParam("days", "Comma-separated weekday names, e.g. 'Monday,Wednesday'."),
                QueryParam("open_only", "'true' to return only classes with open seats."),
            ),
            to_url=_classlist_url,
            needs_js=True,
        ),
        QuerySource(
            key="scholarship_search",
            description=(
                "Search ASU's scholarship database by academic level, citizenship and GPA. Use "
                "when a student asks what they personally qualify for."
            ),
            params=(
                QueryParam("academic_level", "e.g. 'undergrad' or 'graduate'."),
                QueryParam("citizenship", "e.g. 'us_citizen', 'international'."),
                QueryParam("gpa", "Minimum GPA, e.g. '3.0'."),
            ),
            to_url=_scholarship_url,
            needs_js=True,
        ),
    )
}


def url_for(source: QuerySource, params: dict[str, str]) -> str:
    """Validates the parameters, then builds the URL to fetch.

    Raises:
        QueryError: a required parameter is missing, or one was passed that does not exist.
    """
    missing = [p.name for p in source.params if p.required and not params.get(p.name, "").strip()]
    if missing:
        raise QueryError(f"{source.key} needs: {', '.join(missing)}")
    unknown = set(params) - {p.name for p in source.params}
    if unknown:
        raise QueryError(
            f"{source.key} has no parameter {', '.join(sorted(unknown))}; "
            f"it takes: {', '.join(p.name for p in source.params)}"
        )
    return source.to_url(params)
