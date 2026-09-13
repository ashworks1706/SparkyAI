"""The registry of live query sources, the checks every query passes, and how one is run.

Distinct from sources/, which are pages fetched on a schedule into the retrieval index. A query
source is fetched when the model asks, with the parameters it gives, and answers that caller.
"""

from __future__ import annotations

from scraper.core.types import QueryError, QuerySource
from scraper.ingest import extract, fetch
from scraper.query.sources import (
    campus_map,
    clubs,
    courses,
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
)

_MODULES = (
    courses,
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
    jobs,
)

QUERY_SOURCES: dict[str, QuerySource] = {m.QUERY.key: m.QUERY for m in _MODULES}


def check(source: QuerySource, params: dict[str, str]) -> None:
    """Raises QueryError when a required parameter is missing, one does not exist, or a value
    is not among the choices of its parameter."""
    missing = [p.name for p in source.params if p.required and not params.get(p.name, "").strip()]
    if missing:
        raise QueryError(f"{source.key} needs: {', '.join(missing)}")
    unknown = set(params) - {p.name for p in source.params}
    if unknown:
        taken = ", ".join(p.name for p in source.params) or "nothing"
        raise QueryError(
            f"{source.key} has no parameter {', '.join(sorted(unknown))}; it takes: {taken}"
        )
    for p in source.params:
        value = params.get(p.name, "").strip()
        if not p.choices or not value:
            continue
        values = [v.strip() for v in value.split(",") if v.strip()] if p.many else [value]
        if not p.many and "," in value:
            raise QueryError(f"{p.name} takes one value, got {value!r}")
        allowed = {c.lower() for c in p.choices}
        for v in values:
            if v.lower() not in allowed:
                raise QueryError(f"{p.name} {v!r} is not one of: {', '.join(p.choices)}")


def url_for(source: QuerySource, params: dict[str, str]) -> str:
    """Checks the parameters, then builds the URL of a page source."""
    check(source, params)
    if source.to_url is None:
        raise QueryError(f"{source.key} is not fetched from one page")
    return source.to_url(params)


def run(source: QuerySource, params: dict[str, str]) -> tuple[str, str]:
    """Checks the parameters and fetches the source. Returns the URL to cite and the text."""
    check(source, params)
    if source.answer is not None:
        return source.answer(params)
    url = url_for(source, params)
    fetched = fetch.fetch(url, needs_js=source.needs_js)
    text = source.extractor(fetched) if source.extractor else extract.page_text(fetched)
    return url, text
