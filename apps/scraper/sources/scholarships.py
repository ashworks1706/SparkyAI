"""Source: scholarships."""

from scraper.sources import Source

SOURCE = Source(
    key="scholarships",
    url="https://scholarships.asu.edu/",
    category="scholarships",
    fetch_every_hours=72,
    needs_js=False,
)
