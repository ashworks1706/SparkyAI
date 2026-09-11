"""Shuttles: each route kept as one line carrying its name, stops and running times."""

from __future__ import annotations

from scraper.core.types import Fetched, Source
from scraper.extract import blocks, flatten_tables, listing_lines, page_text, plain

_LABELS = {
    "Stops": ("stop", "serves", "route stops"),
    "Runs": ("runs", "running", "schedule", "hours", "service", "frequency", "departs"),
}


def extract_shuttles(fetched: Fetched) -> str:
    """One line per route: name, stops, running times."""
    out: list[str] = []
    text = "\n".join(flatten_tables(page_text(fetched).splitlines()))
    for name, raw_lines in blocks(text):
        body = [line for line in (plain(raw) for raw in raw_lines) if line]
        if name is None:
            out.extend(body)
            continue
        out.extend(listing_lines(name, body, _LABELS))
    return "\n".join(out).strip()


SOURCE = Source(
    key="shuttles",
    url="https://cfo.asu.edu/shuttles",
    category="transit",
    fetch_every_hours=168,
    needs_js=False,
    extractor=extract_shuttles,
)
