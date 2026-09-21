"""Sun Devil Central listings read from the rendered DOM: one line per club or event."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from scraper.core.types import Fetched
from scraper.ingest.extract import form_page_text

BASE = "https://sundevilcentral.eoss.asu.edu"
_ABOUT_CHARS = 400
_SPACE = re.compile(r"\s+")


def _clean(node: Tag | None) -> str:
    return _SPACE.sub(" ", node.get_text(" ", strip=True)).strip() if node else ""


def _clip(text: str, limit: int = _ABOUT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _count(soup: BeautifulSoup, pattern: str) -> str:
    """The total a listing heading reports, such as Events (3025), or empty."""
    found = re.search(pattern, soup.get_text(" ", strip=True))
    return found.group(1) if found else ""


_FILLER = frozenset(
    "a an and any asu at campus central club clubs devil devils event events for group groups in "
    "near next of on org sun sundevil "
    "orgs organization organizations student students the this today tomorrow tonight "
    "upcoming week weekend".split()
)


def keywords(raw: str) -> str:
    """The search words Sun Devil Central can match as one phrase. All filler is no search."""
    return " ".join(w for w in raw.split() if w.lower().strip(".,?!") not in _FILLER)


def extract_clubs(fetched: Fetched) -> str:
    """One entry per group: name, campus and categories, link, website, contact, mission.

    Groups that carry the search as a whole word are kept, those naming it first; when none do,
    every group is kept. A page with no group entries falls back to its text.
    """
    soup = BeautifulSoup(fetched.body, "lxml")
    entries = [e for e in (_club(item) for item in soup.select("li.list-group-item")) if e]
    total = _count(soup, r"Groups\s*\((\d+)\)")
    if not entries and total:
        return f"No groups match this search on Sun Devil Central ({total} groups in total)."
    if not entries:
        return form_page_text(fetched)
    search = _search_term(fetched.url, "search")
    shown, note = _rank(entries, search)
    head = f"{len(shown)} matching groups" + (f" of {total}" if total else "")
    if search:
        head += f" for {search}"
    return "\n\n".join(x for x in (head + note, *(text for _name, text in shown)) if x)


def _club(item: Tag) -> tuple[str, str] | None:
    """One group entry as its name and its text, or None for a row that is not a group."""
    box = item.select_one("input[id^=cb_club_]")
    title = item.select_one("h2 a")
    if box is None or title is None:
        return None
    club_id = box.get("id", "").removeprefix("cb_club_")
    name = _clean(title)
    parts = [name, _clean(item.select_one("p.grey-element"))]
    parts.append(f"{BASE}/student_community?club_id={club_id}")
    site = item.select_one("a[aria-label=Website]")
    if site and site.get("href"):
        parts.append(f"website {site['href']}")
    contact = item.select_one("a[title^='Send a Message to']")
    if contact:
        parts.append(f"contact {_clean(contact)}")
    line = " | ".join(p for p in parts if p)
    mission = _about(item.select_one(f"#club_{club_id}"), "Mission")
    benefits = _about(item.select_one(f"#club_whatwedo_{club_id}"), "Membership Benefits")
    return name, "\n".join(x for x in (line, mission, benefits) if x)


def _search_term(url: str, param: str) -> str:
    values = parse_qs(urlsplit(url).query).get(param, [])
    return values[0].strip() if values else ""


def _rank(entries: list[tuple[str, str]], search: str) -> tuple[list[tuple[str, str]], str]:
    """Entries holding every search word whole, name matches first, and a note on what was cut."""
    words = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in search.split()]
    if not words:
        return entries, ""
    in_name = [e for e in entries if all(w.search(e[0]) for w in words)]
    in_text = [e for e in entries if e not in in_name and all(w.search(e[1]) for w in words)]
    kept = in_name + in_text
    if not kept:
        return entries, " (none carry the words whole; partial matches follow)"
    cut = len(entries) - len(kept)
    return kept, f" ({cut} partial-word matches left out)" if cut else ""


def _about(node: Tag | None, label: str) -> str:
    text = _clean(node).removeprefix(label).strip()
    return f"{label}: {_clip(text)}" if text else ""


def extract_events(fetched: Fetched) -> str:
    """One line per event: name, when, where, attendance, tags, and its page.

    A page with no event entries falls back to its text.
    """
    soup = BeautifulSoup(fetched.body, "lxml")
    out: list[str] = []
    for item in soup.select("li.list-group-item[id^=event_]"):
        title = item.select_one("h3 a")
        if title is None:
            continue
        event_id = item["id"].removeprefix("event_")
        when = _clean(item.select_one(".col-md-5"))
        where = _clean(item.select_one(".col-md-4"))
        going = _clean(item.select_one(".event_display1"))
        price = _clean(item.select_one(".img-label"))
        tags = [_clean(t) for t in item.select(".rsvp__event-tags .label-tag")]
        parts = [_clean(title), when, where, going, price]
        if tags:
            parts.append("tags " + ", ".join(dict.fromkeys(t for t in tags if t)))
        parts.append(urljoin(BASE, f"/rsvp_boot?id={event_id}"))
        out.append(" | ".join(p for p in parts if p))
    total = _count(soup, r"Events\s*\((\d+)\)")
    if not out and total:
        return "No upcoming Sun Devil Central events match this search."
    if not out:
        return form_page_text(fetched)
    head = f"{len(out)} upcoming events shown" + (f" of {total}" if total else "")
    return head + "\n" + "\n".join(out)
