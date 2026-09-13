"""Sun Devil Athletics schedules and results for one sport."""

from __future__ import annotations

from scraper.core.types import QueryParam, QuerySource
from scraper.query.params import choices, code

# Path segment of each sport on thesundevils.com.
_SPORTS = {
    "football": "football",
    "men's basketball": "mens-basketball",
    "women's basketball": "womens-basketball",
    "baseball": "baseball",
    "softball": "softball",
    "volleyball": "womens-volleyball",
    "soccer": "womens-soccer",
    "hockey": "mens-ice-hockey",
    "wrestling": "wrestling",
}


def to_url(params: dict[str, str]) -> str:
    """The schedule page of the sport."""
    return f"https://thesundevils.com/sports/{code(params, 'sport', _SPORTS)}/schedule"


QUERY = QuerySource(
    key="sports",
    description="Fetch the Sun Devil schedule and recent results for one sport.",
    params=(QueryParam("sport", "Sport.", required=True, choices=choices(_SPORTS)),),
    to_url=to_url,
    needs_js=True,
)
