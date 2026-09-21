"""ASU events: the public events calendar and the login-gated Sun Devil Central listings, merged."""

from __future__ import annotations

import re
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup, Tag

from scraper.core.settings import settings
from scraper.core.types import AuthError, Fetched, QueryError, QueryParam, QuerySource
from scraper.ingest import fetch
from scraper.query import sundevil_central
from scraper.query.params import text, url
from scraper.sources.events import extract_events

log = structlog.get_logger()

_PUBLIC = "https://asuevents.asu.edu/home"
_SUNDEVIL = "https://sundevilcentral.eoss.asu.edu/events"


_SPACE = re.compile(r"\s+")


def _clean(node: Tag | None) -> str:
    return _SPACE.sub(" ", node.get_text(" ", strip=True)).strip() if node else ""


def public_cards(fetched: Fetched) -> str:
    """One line per event card on the public calendar: name, date, time, place, and its page.

    A page without cards goes through the line-based extractor.
    """
    if fetched.text is not None:
        return extract_events(fetched)
    out: list[str] = []
    for card in BeautifulSoup(fetched.body, "lxml").select("li.card-event"):
        title = card.select_one("h3.card-title a")
        if title is None:
            continue
        parts = [
            _clean(title),
            _clean(card.select_one(".views-field-field-event-date-value-1")),
            _clean(card.select_one(".views-field-field-event-date-end-value")),
            _clean(card.select_one(".views-field-field-asu-events-location")),
            urljoin(_PUBLIC, title.get("href", "").split("&")[0]),
        ]
        out.append(" | ".join(p for p in parts if p))
    return "\n".join(out) if out else extract_events(fetched)


def _public(keywords: str) -> tuple[str, str]:
    """The public ASU events calendar, narrowed to a keyword when one was given."""
    target = url(_PUBLIC, [("searchText", keywords)])
    return target, public_cards(fetch.fetch(target, needs_js=True))


def _sundevil(keywords: str) -> tuple[str, str]:
    """The login-gated Sun Devil Central events, read through the admin session."""
    target = url(_SUNDEVIL, [("search_word", sundevil_central.keywords(keywords))])
    return target, sundevil_central.extract_events(fetch.fetch(target, auth=True))


def _fit(body: str, budget: int) -> str:
    """body cut at a line boundary to fit budget characters, so every section reaches the model."""
    if len(body) <= budget:
        return body
    kept = body[:budget].rsplit("\n", 1)[0]
    return f"{kept}\n[more listings not shown]"


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Both event listings combined. Missing or expired auth drops the Sun Devil Central section."""
    keywords = text(params, "keywords")
    sections: list[str] = []
    cite: str | None = None
    readers = (("Sun Devil Central", _sundevil), ("ASU Events", _public))
    budget = settings().scraper.query_max_chars // len(readers)
    for label, reader in readers:
        try:
            source_url, body = reader(keywords)
        except AuthError as e:
            log.warning("sun devil central events skipped", error=str(e))
            continue
        except Exception as e:  # one unreadable listing leaves the other
            log.warning("event source failed", section=label, error=str(e))
            continue
        if body.strip():
            sections.append(f"{label}\n{_fit(body.strip(), budget)}")
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
