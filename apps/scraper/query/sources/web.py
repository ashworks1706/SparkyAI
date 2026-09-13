"""Web search through SearXNG: titles, links and snippets from Google, Brave and Bing."""

from __future__ import annotations

import urllib.parse
from typing import Any

from scraper.core.settings import settings
from scraper.core.types import QueryError, QueryParam, QuerySource
from scraper.query.http import get_json, plain
from scraper.query.params import text

_TIME_RANGES = ("day", "week", "month", "year")


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Asks SearXNG and lists the results. The citation is the Google search for the query."""
    cfg = settings().search
    query = text(params, "query")
    search = {
        "q": query,
        "format": "json",
        "engines": cfg.engines,
        "language": cfg.language,
        "safesearch": str(cfg.safesearch),
    }
    time_range = text(params, "time_range").lower()
    if time_range:
        search["time_range"] = time_range
    found = get_json(f"{cfg.base_url.rstrip('/')}/search?{urllib.parse.urlencode(search)}")
    citation = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
    return citation, render(query, found, cfg.max_results, cfg.snippet_chars)


def render(query: str, found: dict[str, Any], limit: int, snippet_chars: int) -> str:
    """Direct answers first, then one block per result: title, link, date when known, snippet."""
    lines = [f"Web results for {query!r}:"]
    for direct in found.get("answers", [])[:2]:
        body = direct.get("answer") if isinstance(direct, dict) else direct
        if body:
            lines.append(f"Answer: {plain(str(body), snippet_chars)}")
    seen: set[str] = set()
    for result in found.get("results", []):
        url = str(result.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        title = plain(result.get("title"), 160)
        dated = str(result.get("publishedDate") or "")[:10]
        head = f"{len(seen)}. {title} | {url}" + (f" | {dated}" if dated else "")
        lines.append(head)
        snippet = plain(result.get("content"), snippet_chars)
        if snippet:
            lines.append(f"   {snippet}")
        if len(seen) == limit:
            break
    if not seen:
        down = ", ".join(e[0] for e in found.get("unresponsive_engines", []) if e)
        reason = f" (engines not answering: {down})" if down else ""
        raise QueryError(f"the web search for {query!r} found nothing{reason}")
    return "\n".join(lines)


QUERY = QuerySource(
    key="web",
    description=(
        "Search the web through Google, Brave and Bing for anything the ASU sources do not cover."
    ),
    params=(
        QueryParam("query", "What to search for.", required=True, example="ASU vs Texas A&M score"),
        QueryParam("time_range", "Only results from this recent period.", choices=_TIME_RANGES),
    ),
    answer=answer,
    category="web",
    index=False,
)
