"""One module per public ASU source, registered in SOURCES. A source is a row in `sources`."""

from __future__ import annotations

from scraper.core.types import Source
from scraper.sources import (
    clubs,
    courses,
    events,
    jobs,
    library_hours,
    news,
    scholarships,
    shuttles,
    sports,
)

SOURCES: dict[str, Source] = {
    m.SOURCE.key: m.SOURCE
    for m in (library_hours, events, clubs, courses, scholarships, news, shuttles, jobs, sports)
}
