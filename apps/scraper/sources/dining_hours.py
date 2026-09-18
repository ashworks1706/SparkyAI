"""Dining hours: the per-campus PDFs, one line per venue pairing each day with its hours."""

from __future__ import annotations

import html

from scraper.core.types import Fetched, Source
from scraper.ingest.extract import page_text

_TIME_MARKS = ("a.m.", "p.m.", "noon", "midnight")


def extract_dining_hours(fetched: Fetched) -> str:
    """One line per venue. The PDF lists a venue's day labels first and their hours after."""
    out: list[str] = []
    name: str | None = None
    days: list[str] = []
    hours: list[str] = []

    def flush() -> None:
        if name is None:
            return
        if not days:
            out.append(name)
            return
        pairs = "; ".join(f"{day} {time}" for day, time in zip(days, hours, strict=False))
        out.append(f"{name}: {pairs}")

    for raw in page_text(fetched).splitlines():
        line = " ".join(html.unescape(raw).split())
        if not line or line == "*":
            continue
        if line.endswith(":") and not hours:
            days.append(line[:-1].strip())
        elif days and _is_hours(line):
            hours.append(line)
        else:
            flush()
            name, days, hours = line, [], []
    flush()
    return "\n".join(out).strip()


def _is_hours(line: str) -> bool:
    lower = line.lower()
    return lower == "closed" or any(mark in lower for mark in _TIME_MARKS)


_PDFS = "https://sundevilhospitality.asu.edu/sites/g/files/litvpz701/files"

#: One source per campus. The PDF path changes each term; the campus page under
#: sundevilhospitality.asu.edu/hours-locations links the current one.
SOURCES: tuple[Source, ...] = tuple(
    Source(
        key=f"dining_hours_{campus}",
        url=f"{_PDFS}/{path}",
        category="dining",
        fetch_every_hours=72,
        extractor=extract_dining_hours,
    )
    for campus, path in (
        ("tempe", "2026-08/0826-Fall-Hours-Tempe%20%281%29.pdf"),
        ("downtown", "2026-09/0726-Fall-Hours-Downtown.pdf"),
        ("west", "2026-08/0826-Fall-Hours-West.pdf"),
    )
)
