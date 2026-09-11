"""Scholarships: each award kept as one line carrying its name, amount, deadline and eligibility."""

from __future__ import annotations

from scraper.core.types import Fetched, Source
from scraper.extract import blocks, flatten_tables, labeled, listing_lines, page_text, plain

_LABELS = {
    "Award": ("award", "amount", "value", "stipend"),
    "Deadline": ("deadline", "closes", "closing", "due", "apply by"),
    "Eligibility": ("eligib", "open to", "who can apply", "requirements"),
}
_ELIGIBILITY_OPENERS = ("open to", "available to", "for students", "awarded to")


def extract_scholarships(fetched: Fetched) -> str:
    """One line per scholarship: name, award, deadline, eligibility."""
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
    if labeled(line) is not None:
        return line
    if line.startswith("$"):
        return f"Award: {line}"
    if line.lower().startswith(_ELIGIBILITY_OPENERS):
        return f"Eligibility: {line}"
    return line


SOURCE = Source(
    key="scholarships",
    url="https://scholarships.asu.edu/",
    category="scholarships",
    fetch_every_hours=72,
    needs_js=False,
    extractor=extract_scholarships,
)
