"""Events: each listing kept as one line carrying its name, date and location."""

from __future__ import annotations

import re

from scraper.core.types import Fetched, Source
from scraper.extract import page_text, plain

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE = re.compile(rf"\b(?:{_MONTHS})\b\.?\s+\d{{1,2}}|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b")
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)", re.IGNORECASE)
_HEADING = re.compile(r"^\s*#{1,6}\s")
_LINK_ONLY = re.compile(r"^\s*[-*+]?\s*\[[^\]]+\]\([^)]*\)\s*$")
_PLACE = re.compile(
    r"\b(campus|hall|room|center|centre|library|building|auditorium|field|stadium"
    r"|online|virtual|zoom|lawn|union|gym|theatre|theater|arena|plaza)\b",
    re.IGNORECASE,
)
_MAX_PLACE_CHARS = 120


def extract_events(fetched: Fetched) -> str:
    """One line per event: name, when, where. Anything else on the page follows it."""
    out: list[str] = []
    event: _Event | None = None
    for raw in page_text(fetched).splitlines():
        line = plain(raw)
        if not line:
            continue
        if _starts_event(raw):
            _flush(event, out)
            event = _Event(line)
            continue
        if event is None:
            out.append(line)
            continue
        event.add(line)
    _flush(event, out)
    return "\n".join(out).strip()


class _Event:
    """One listing being collected, line by line."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.when: str | None = None
        self.where: str | None = None
        self.rest: list[str] = []
        self._after_when = False

    def add(self, line: str) -> None:
        """Files one line under when, where, or the leftover description."""
        if self.when is None and (_DATE.search(line) or _TIME.search(line)):
            self.when = line
            self._after_when = True
            return
        if self.where is None and len(line) <= _MAX_PLACE_CHARS:
            if self._after_when or _PLACE.search(line):
                self.where = line
                self._after_when = False
                return
        self._after_when = False
        self.rest.append(line)

    def lines(self) -> list[str]:
        """The event as a headline plus whatever did not fit on it."""
        parts = [part for part in (self.name, self.when, self.where) if part]
        return [" | ".join(parts), *self.rest]


def _starts_event(raw: str) -> bool:
    return bool(_HEADING.match(raw) or _LINK_ONLY.match(raw))


def _flush(event: _Event | None, out: list[str]) -> None:
    if event is not None:
        out.extend(event.lines())


SOURCE = Source(
    key="events",
    url="https://asuevents.asu.edu/",
    category="events",
    fetch_every_hours=12,
    needs_js=True,
    extractor=extract_events,
)
