"""Sports: each fixture kept as one line carrying its opponent, date and venue."""

from __future__ import annotations

from scraper.core.types import Fetched, Source
from scraper.extract import blocks, flatten_tables, listing_lines, page_text, plain

_LABELS = {
    "Opponent": ("opponent", "versus", "vs", "against"),
    "Date": ("date", "time", "kickoff", "first pitch", "tip"),
    "Location": ("location", "venue", "site", "stadium", "arena"),
}


def extract_sports(fetched: Fetched) -> str:
    """One line per fixture: event, opponent, date, venue."""
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
    key="sports",
    url="https://thesundevils.com/",
    category="sports",
    fetch_every_hours=24,
    needs_js=True,
    extractor=extract_sports,
)
