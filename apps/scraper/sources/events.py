"""Source: events."""

from scraper.sources import Source

SOURCE = Source(
    key="events",
    url="https://asuevents.asu.edu/",
    category="events",
    fetch_every_hours=12,
    needs_js=True,
)
