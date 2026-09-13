"""Sun Devil Athletics news for one sport, and outside coverage of it."""

from __future__ import annotations

import email.utils
import urllib.parse
import xml.etree.ElementTree as ET

from scraper.core.types import QueryParam, QuerySource
from scraper.query.http import day, get_xml, plain
from scraper.query.params import text

NEWS = "https://thesundevils.com/news"
_OFFICIAL = "https://thesundevils.com/rss"
_COVERAGE = "https://news.google.com/rss/search"
_MAX_OFFICIAL = 8
_MAX_COVERAGE = 6

# The sport names the official feed tags its stories with.
_SPORTS = (
    "football",
    "men's basketball",
    "women's basketball",
    "baseball",
    "softball",
    "volleyball",
    "beach volleyball",
    "soccer",
    "ice hockey",
    "wrestling",
    "gymnastics",
    "men's golf",
    "women's golf",
    "cross country",
    "track and field",
    "triathlon",
    "men's tennis",
    "women's tennis",
    "men's swimming and diving",
    "women's swimming and diving",
)


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Reads the official news feed and a news search feed."""
    sport = text(params, "sport").lower()
    keywords = text(params, "keywords")
    query = " ".join(p for p in ('"Arizona State"', sport or "Sun Devils", keywords) if p)
    search = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    coverage_url = f"{_COVERAGE}?{urllib.parse.urlencode(search)}"
    official = official_lines(get_xml(_OFFICIAL), sport, keywords)
    coverage = coverage_lines(get_xml(coverage_url))
    parts = ["Sun Devil Athletics news:", *(official or ["No matching stories."])]
    parts += ["", "Other coverage:", *(coverage or ["No matching stories."])]
    return NEWS, "\n".join(parts)


def official_lines(feed: ET.Element, sport: str, keywords: str) -> list[str]:
    """Official stories, newest first, narrowed to the sport tag and the keywords."""
    lines = []
    for item in feed.iter("item"):
        tag = (item.findtext("sports") or "").strip().lower()
        title = plain(item.findtext("title"), 160)
        if sport and tag != sport:
            continue
        if keywords and keywords.lower() not in title.lower():
            continue
        lines.append(f"{_when(item)} | {title} | {item.findtext('link') or ''}")
        if len(lines) == _MAX_OFFICIAL:
            break
    return lines


def coverage_lines(feed: ET.Element) -> list[str]:
    """Outside stories as the search feed orders them, each with its publisher."""
    lines = []
    for item in list(feed.iter("item"))[:_MAX_COVERAGE]:
        title = plain(item.findtext("title"), 160)
        source = plain(item.findtext("source"), 60)
        lines.append(f"{_when(item)} | {title} | {source}")
    return lines


def _when(item: ET.Element) -> str:
    raw = item.findtext("pubDate")
    if not raw:
        return "undated"
    try:
        return day(email.utils.parsedate_to_datetime(raw))
    except (TypeError, ValueError):
        return "undated"


QUERY = QuerySource(
    key="sports_news",
    description="Latest Sun Devil Athletics news, and outside coverage, for one sport or all.",
    params=(
        QueryParam("sport", "Sport.", choices=_SPORTS),
        QueryParam("keywords", "Words in the headline.", example="Texas A&M"),
    ),
    answer=answer,
)
