"""One module per public ASU source, registered in SOURCES.

A source is a row in `sources`, never a folder. Port order per docs/ROADMAP.md Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    category: str
    fetch_every_hours: int = 24
    needs_js: bool = False


def _load() -> dict[str, Source]:
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

    modules = (library_hours, events, clubs, courses, scholarships, news, shuttles, jobs, sports)
    return {m.SOURCE.key: m.SOURCE for m in modules}


SOURCES: dict[str, Source] = _load()
