"""ASU events: the public events calendar and the login-gated Sun Devil Central listings, merged."""

from __future__ import annotations

import structlog

from scraper.core.types import AuthError, QueryError, QueryParam, QuerySource
from scraper.ingest import fetch
from scraper.ingest.auth import fetch_authenticated
from scraper.query.params import text, url
from scraper.sources.events import extract_events

log = structlog.get_logger()

_PUBLIC = "https://asuevents.asu.edu/home"
_SUNDEVIL = "https://sundevilcentral.eoss.asu.edu/events"


def _public(keywords: str) -> tuple[str, str]:
    """The public ASU events calendar, narrowed to a keyword when one was given."""
    target = url(_PUBLIC, [("searchText", keywords)])
    return target, extract_events(fetch.fetch(target, needs_js=True))


def _sundevil(keywords: str) -> tuple[str, str]:
    """The login-gated Sun Devil Central events, read through the admin session."""
    target = url(_SUNDEVIL, [("search", keywords)])
    return target, extract_events(fetch_authenticated(target))


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Both event listings combined. Missing or expired auth drops the Sun Devil Central section."""
    keywords = text(params, "keywords")
    sections: list[str] = []
    cite: str | None = None
    for label, reader in (("ASU Events", _public), ("Sun Devil Central", _sundevil)):
        try:
            source_url, body = reader(keywords)
        except AuthError as e:
            log.warning("sun devil central events skipped", error=str(e))
            continue
        except (QueryError, RuntimeError) as e:
            log.warning("event source failed", section=label, error=str(e))
            continue
        if body.strip():
            sections.append(f"{label}\n{body.strip()}")
            cite = cite or source_url
    if not sections:
        raise QueryError("no events found and the events sources could not be read")
    return cite or _PUBLIC, "\n\n".join(sections)


QUERY = QuerySource(
    key="events",
    description="Search ASU events: the public calendar and Sun Devil Central listings.",
    params=(QueryParam("keywords", "What the event is about.", example="career fair"),),
    answer=answer,
    category="events",
    auth=True,
    index=False,
)
