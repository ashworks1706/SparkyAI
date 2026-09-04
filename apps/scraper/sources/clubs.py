"""Source: clubs."""

from scraper.sources import Source

SOURCE = Source(
    key="clubs",
    url="https://asu.campuslabs.com/engage/organizations",
    category="clubs",
    fetch_every_hours=168,
    needs_js=True,
)
