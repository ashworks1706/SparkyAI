"""Source: shuttles."""

from scraper.sources import Source

SOURCE = Source(
    key="shuttles",
    url="https://cfo.asu.edu/shuttles",
    category="transit",
    fetch_every_hours=168,
    needs_js=False,
)
