"""The ASU course catalog: the description, credit hours and prerequisites of a course."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import term_code, text, url


def to_url(params: dict[str, str]) -> str:
    """The course list, narrowed to a keyword and, when one was given, to a term."""
    term = text(params, "term")
    return url(
        "https://catalog.apps.asu.edu/catalog/courses/courselist",
        [
            ("keywords", text(params, "keywords")),
            ("term", term_code(term) if term else ""),
            ("advanced", "false"),
        ],
    )


QUERY = QuerySource(
    key="course_catalog",
    description=(
        "Look one ASU course up in the catalog: what it covers, how many credit hours it "
        "carries, and the prerequisites it lists. It holds no sections and no seat counts."
    ),
    params=(
        QueryParam("keywords", "Course code, subject or title.", required=True, example="CSE 485"),
        QueryParam("term", "Term the catalog entry is read for.", example="Fall 2026"),
    ),
    to_url=to_url,
    needs_js=True,
    category="course_catalog",
)
