"""Source: news."""

from scraper.sources import Source

SOURCE = Source(
    key="news",
    url="https://news.asu.edu/",
    category="news",
    fetch_every_hours=12,
    needs_js=False,
)
