"""The registry of live query sources, keyed by the name the engine calls them by."""

from __future__ import annotations

from scraper.core.types import QuerySource
from scraper.query.sources import (
    campus_map,
    clubs,
    course_catalog,
    courses,
    dining,
    events,
    jobs,
    library_catalog,
    library_hours,
    news,
    scholarships,
    shuttles,
    social_media,
    sports,
    sports_news,
    study_rooms,
    web,
)

_MODULES = (
    courses,
    course_catalog,
    scholarships,
    events,
    clubs,
    news,
    library_catalog,
    library_hours,
    study_rooms,
    sports,
    sports_news,
    shuttles,
    campus_map,
    social_media,
    dining,
    jobs,
    web,
)

QUERY_SOURCES: dict[str, QuerySource] = {m.QUERY.key: m.QUERY for m in _MODULES}
