"""Live next-bus times at every stop of the ASU intercampus shuttles, from the shuttle tracker."""

from __future__ import annotations

import datetime
import re
from typing import Any

from scraper.core.types import QueryError, QueryParam, QuerySource
from scraper.query.http import ARIZONA, clock, get_json, get_text
from scraper.query.params import text

TRACKER = "https://asu-shuttles.rider.peaktransit.com/"
_CONFIG = "https://asu-shuttles.rider.peaktransit.com/configs/asu-shuttles.js"
_API = re.compile(r'BPTAPI\s*=\s*"([^"]+)"')
_AGENCY = re.compile(r"agencyID\s*=\s*(\d+)")

# Choice to the long route name the tracker uses.
_ROUTES = {
    "mercado": "Mercado",
    "polytechnic-tempe": "Polytechnic-Tempe",
    "tempe-downtown phoenix-west": "Tempe-Downtown Phoenix-West",
    "tempe-west express": "Tempe-West Express",
}


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Reads the tracker API: its key from the tracker page, then routes, stops and ETAs."""
    found = _API.search(get_text(TRACKER))
    agency = _AGENCY.search(get_text(_CONFIG))
    if found is None or agency is None:
        raise QueryError("the shuttle tracker no longer publishes its API; try again later")
    base = f"{found.group(1)}&agencyID={agency.group(1)}"
    routes = get_json(f"{base}&controller=route2&action=list").get("routes", [])
    stops = get_json(f"{base}&controller=stop2&action=list").get("stop", [])
    etas = get_json(f"{base}&controller=eta&action=list").get("stop", [])
    wanted = _ROUTES.get(text(params, "route").lower())
    now = datetime.datetime.now(ARIZONA)
    return TRACKER, render(routes, stops, etas, wanted, now)


def render(
    routes: list[dict[str, Any]],
    stops: list[dict[str, Any]],
    etas: list[dict[str, Any]],
    wanted: str | None,
    now: datetime.datetime,
) -> str:
    """One block per route in service: every stop with its next two buses, soonest first."""
    names = {s["stopID"]: s.get("longName", "") for s in stops}
    lines = [f"Live shuttle times as of {clock(now)} Arizona time."]
    for route in routes:
        name = route.get("longName") or route.get("shortName", "")
        if route.get("hidden") or route.get("disabled") or (wanted and name != wanted):
            continue
        lines.append(f"\nRoute: {name}")
        upcoming = [e for e in etas if e.get("routeID") == route.get("routeID")]
        upcoming.sort(key=lambda e: e.get("ETA1") or float("inf"))
        if not upcoming:
            lines.append("No bus is scheduled on this route now.")
        for eta in upcoming:
            times = [_at(eta.get(k)) for k in ("ETA1", "ETA2")]
            shown = ", then ".join(t for t in times if t) or "no bus scheduled"
            lines.append(f"{names.get(eta.get('stopID'), 'Unnamed stop')}: next {shown}")
    if len(lines) == 1:
        lines.append("No route matched.")
    return "\n".join(lines)


def _at(epoch: int | None) -> str:
    return clock(datetime.datetime.fromtimestamp(epoch, ARIZONA)) if epoch else ""


QUERY = QuerySource(
    key="shuttles",
    description="Live next-bus times at every stop of the ASU intercampus shuttles.",
    params=(QueryParam("route", "Shuttle route.", choices=tuple(_ROUTES)),),
    answer=answer,
)
