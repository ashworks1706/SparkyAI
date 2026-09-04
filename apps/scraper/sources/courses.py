from scraper.core.types import Source

SOURCE = Source(
    key="courses",
    url="https://catalog.asu.edu/",
    category="courses",
    fetch_every_hours=168,
    needs_js=False,
)
