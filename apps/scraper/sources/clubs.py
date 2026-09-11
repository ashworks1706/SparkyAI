"""Clubs: each organization kept as one line carrying its name, category and description."""

from __future__ import annotations

from scraper.core.types import Fetched, Source
from scraper.extract import blocks, labeled, page_text, plain

_CATEGORY_LABELS = ("category", "type", "interest")
_CATEGORY_CHARS = 60
_CATEGORY_WORDS = 6


def extract_clubs(fetched: Fetched) -> str:
    """One line per club: name, category, description. Page prose is kept as it reads."""
    out: list[str] = []
    for name, raw_lines in blocks(page_text(fetched)):
        body = [line for line in (plain(raw) for raw in raw_lines) if line]
        if name is None:
            out.extend(body)
            continue
        out.extend(_club_lines(name, body))
    return "\n".join(out).strip()


def _club_lines(name: str, body: list[str]) -> list[str]:
    category: str | None = None
    rest: list[str] = []
    for line in body:
        pair = labeled(line)
        if category is None and pair and pair[0].lower().startswith(_CATEGORY_LABELS):
            category = pair[1]
            continue
        if category is None and pair is None and _is_category(line):
            category = line
            continue
        rest.append(line)
    description = rest[0] if rest else None
    parts = [part for part in (name, category, description) if part]
    return [" | ".join(parts), *rest[1:]]


def _is_category(line: str) -> bool:
    return len(line) <= _CATEGORY_CHARS and len(line.split()) <= _CATEGORY_WORDS


SOURCE = Source(
    key="clubs",
    url="https://asu.campuslabs.com/engage/organizations",
    category="clubs",
    fetch_every_hours=168,
    needs_js=True,
    extractor=extract_clubs,
)
