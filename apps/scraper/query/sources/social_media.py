"""Latest posts on the official ASU and Sun Devil Athletics YouTube channels."""

from __future__ import annotations

import datetime
import xml.etree.ElementTree as ET

from scraper.core.types import QueryParam, QuerySource
from scraper.query.http import day, get_xml, plain
from scraper.query.params import text

_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id="
_NS = {"a": "http://www.w3.org/2005/Atom", "m": "http://search.yahoo.com/mrss/"}
_MAX_POSTS = 8
_DESCRIPTION_CHARS = 160

# Choice to the channel name and its YouTube channel id.
_ACCOUNTS = {
    "asu": ("Arizona State University", "UC027E0kdtyo0h4lLGROg-dA"),
    "sun devil athletics": ("Sun Devil Athletics", "UC_G-_2azcOV9JBAmqFgERwA"),
}


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Reads the feed of each account asked for, or of every account."""
    asked = text(params, "account").lower()
    keywords = text(params, "keywords")
    accounts = [asked] if asked else list(_ACCOUNTS)
    blocks = []
    for key in accounts:
        name, channel = _ACCOUNTS[key]
        posts = posts_of(get_xml(_FEED + channel), keywords)
        blocks.append("\n".join([f"{name} on YouTube:", *(posts or ["No matching posts."])]))
    first = _ACCOUNTS[accounts[0]][1]
    return f"https://www.youtube.com/channel/{first}", "\n\n".join(blocks)


def posts_of(feed: ET.Element, keywords: str) -> list[str]:
    """The newest posts of a channel feed, narrowed to the keywords."""
    lines = []
    for entry in feed.findall("a:entry", _NS):
        title = plain(entry.findtext("a:title", namespaces=_NS), 160)
        about = plain(entry.findtext("m:group/m:description", namespaces=_NS), _DESCRIPTION_CHARS)
        if keywords and keywords.lower() not in f"{title} {about}".lower():
            continue
        link = entry.find("a:link", _NS)
        href = link.get("href", "") if link is not None else ""
        lines.append(f"{_published(entry)} | {title} | {href} | {about}")
        if len(lines) == _MAX_POSTS:
            break
    return lines


def _published(entry: ET.Element) -> str:
    raw = entry.findtext("a:published", namespaces=_NS)
    try:
        return day(datetime.datetime.fromisoformat(raw)) if raw else "undated"
    except ValueError:
        return "undated"


QUERY = QuerySource(
    key="social_media",
    description="Latest posts on the official ASU and Sun Devil Athletics YouTube channels.",
    params=(
        QueryParam("account", "Account.", choices=tuple(_ACCOUNTS)),
        QueryParam("keywords", "Words in the post.", example="football"),
    ),
    answer=answer,
    category="social",
)
