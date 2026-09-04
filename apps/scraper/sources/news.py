"""Source: news."""

from scraper.core.types import Source

SOURCE = Source(
    key="news",
    url="https://news.asu.edu/",
    category="news",
    fetch_every_hours=12,
    needs_js=False,
)
