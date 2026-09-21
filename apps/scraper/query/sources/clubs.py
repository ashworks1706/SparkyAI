"""Student organizations on Sun Devil Central, searched by name or topic."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import text, url
from scraper.query.sundevil_central import extract_clubs, keywords


def to_url(params: dict[str, str]) -> str:
    """The organization directory, narrowed to a keyword."""
    return url(
        "https://sundevilcentral.eoss.asu.edu/club_signup",
        [("view", "all"), ("search", keywords(text(params, "keywords")))],
    )


QUERY = QuerySource(
    key="clubs",
    description="Search ASU student organizations and clubs on Sun Devil Central.",
    params=(
        QueryParam(
            "keywords", "Club name or topic.", required=True, example="artificial intelligence"
        ),
    ),
    to_url=to_url,
    needs_js=True,
    extractor=extract_clubs,
    category="clubs",
    auth=True,
    index=False,
)
