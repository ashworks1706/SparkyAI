"""Source: library_hours."""

from scraper.sources import Source

SOURCE = Source(
    key="library_hours",
    url="https://lib.asu.edu/hours",
    category="library",
    fetch_every_hours=6,
    needs_js=True,
)
