"""News: each story kept as one line carrying its headline, date and summary."""

from __future__ import annotations

import re

from scraper.core.types import Fetched, Source
from scraper.ingest.extract import blocks, labeled, page_text, plain

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE = re.compile(rf"\b(?:{_MONTHS})\b\.?\s+\d{{1,2}}|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b")
_DATE_LABELS = ("date", "published", "posted")
_MAX_DATE_CHARS = 60


def extract_news(fetched: Fetched) -> str:
    """One line per story: headline, date, summary. Anything else follows that line."""
    out: list[str] = []
    for name, raw_lines in blocks(page_text(fetched)):
        body = [line for line in (plain(raw) for raw in raw_lines) if line]
        if name is None:
            out.extend(body)
            continue
        out.extend(_story_lines(name, body))
    return "\n".join(out).strip()


def _story_lines(headline: str, body: list[str]) -> list[str]:
    date: str | None = None
    rest: list[str] = []
    for line in body:
        pair = labeled(line)
        if date is None and pair and pair[0].lower().startswith(_DATE_LABELS):
            date = pair[1]
            continue
        if date is None and pair is None and _is_date(line):
            date = line
            continue
        rest.append(line)
    summary = rest[0] if rest else None
    parts = [part for part in (headline, date, summary) if part]
    return [" | ".join(parts), *rest[1:]]


def _is_date(line: str) -> bool:
    return len(line) <= _MAX_DATE_CHARS and bool(_DATE.search(line))


SOURCE = Source(
    key="news",
    url="https://news.asu.edu/",
    category="news",
    fetch_every_hours=12,
    needs_js=False,
    extractor=extract_news,
)
