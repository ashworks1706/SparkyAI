"""Source: sports."""

from scraper.core.types import Source

SOURCE = Source(
    key="sports",
    url="https://thesundevils.com/",
    category="sports",
    fetch_every_hours=24,
    needs_js=True,
)
