"""One module per ASU source with its own extractor, plus the static pages. A source is a row."""

from __future__ import annotations

from scraper.core.types import Source
from scraper.sources import (
    courses,
    dining_hours,
    events,
    jobs,
    library_hours,
    news,
    pages,
    scholarships,
    shuttles,
    sports,
)

# Clubs is login-gated on Sun Devil Central, so it is served live through the admin session and
# never scheduled here; a scheduled fetch would run unauthenticated and index nothing usable.
_MODULES = (
    library_hours,
    events,
    courses,
    scholarships,
    news,
    shuttles,
    jobs,
    sports,
)

_ALL: tuple[Source, ...] = (
    *(m.SOURCE for m in _MODULES),
    *dining_hours.SOURCES,
    *pages.PAGES,
)

SOURCES: dict[str, Source] = {s.key: s for s in _ALL}

if len(SOURCES) != len(_ALL):
    raise ValueError("two sources share a key")
