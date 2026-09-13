"""The ASU scholarship search, filtered by what the student says about themselves."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import choices, code, text, url

# Option values of the exposed filters on onsa.asu.edu/scholarships.
_CITIZENSHIP = {
    "us citizen": "75",
    "us permanent resident": "76",
    "daca/dreamer": "77",
    "international student": "78",
}

_APPLICANTS = {
    "first-year undergrad": "58",
    "second-year undergrad": "59",
    "third-year undergrad": "60",
    "fourth-year+ undergrad": "61",
    "graduate student": "62",
    "undergraduate alumni": "63",
    "graduate alumni": "64",
}

_FOCUS = {
    "business and entrepreneurship": "44",
    "creative and performing arts": "45",
    "environment and sustainability": "46",
    "health and medicine": "47",
    "humanities": "48",
    "international affairs": "49",
    "journalism and media": "50",
    "national security": "51",
    "public policy": "53",
    "public service": "52",
    "social justice": "54",
    "social science": "57",
    "stem": "55",
    "peace and conflict resolution": "56",
}


def to_url(params: dict[str, str]) -> str:
    """The scholarship search URL with every filter that was given."""
    return url(
        "https://onsa.asu.edu/scholarships",
        [
            ("combine", text(params, "keywords")),
            ("field_citizenship_status", code(params, "citizenship", _CITIZENSHIP)),
            ("field_eligible_applicants", code(params, "applicant", _APPLICANTS)),
            ("field_focus", code(params, "focus", _FOCUS)),
        ],
    )


QUERY = QuerySource(
    key="scholarships",
    description=(
        "Search ASU scholarships by keyword, citizenship, year of study and field of focus."
    ),
    params=(
        QueryParam("keywords", "Words in the scholarship name or description.", example="women"),
        QueryParam("citizenship", "Citizenship status.", choices=choices(_CITIZENSHIP)),
        QueryParam("applicant", "Year of study.", choices=choices(_APPLICANTS)),
        QueryParam("focus", "Field of focus.", choices=choices(_FOCUS)),
    ),
    to_url=to_url,
)
