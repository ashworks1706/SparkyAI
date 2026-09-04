"""Source: jobs."""

from scraper.core.types import Source

SOURCE = Source(
    key="jobs",
    url="https://students.asu.edu/employment",
    category="jobs",
    fetch_every_hours=48,
    needs_js=False,
)
