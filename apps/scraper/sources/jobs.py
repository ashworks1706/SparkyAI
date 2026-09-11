"""Jobs: each posting kept as one line carrying its title, department, pay and closing date."""

from __future__ import annotations

import re

from scraper.core.types import Fetched, Source
from scraper.extract import blocks, flatten_tables, labeled, listing_lines, page_text, plain

_LABELS = {
    "Department": ("department", "unit", "college", "school", "employer"),
    "Pay": ("pay", "wage", "rate", "salary", "compensation"),
    "Closes": ("closes", "closing", "deadline", "apply by", "open until"),
}
_PAY = re.compile(r"^\$\d")


def extract_jobs(fetched: Fetched) -> str:
    """One line per posting: title, department, pay, closing date."""
    out: list[str] = []
    text = "\n".join(flatten_tables(page_text(fetched).splitlines()))
    for name, raw_lines in blocks(text):
        body = [_relabel(line) for line in (plain(raw) for raw in raw_lines) if line]
        if name is None:
            out.extend(body)
            continue
        out.extend(listing_lines(name, body, _LABELS))
    return "\n".join(out).strip()


def _relabel(line: str) -> str:
    if labeled(line) is None and _PAY.match(line):
        return f"Pay: {line}"
    return line


SOURCE = Source(
    key="jobs",
    url="https://students.asu.edu/employment",
    category="jobs",
    fetch_every_hours=48,
    needs_js=False,
    extractor=extract_jobs,
)
