"""Buildings and places on the ASU campus map, found by name, with a map link."""

from __future__ import annotations

from typing import Any

from scraper.core.types import QueryError, QueryParam, QuerySource
from scraper.query.http import get_json, plain
from scraper.query.params import text, url

MAP = "https://www.asu.edu/about/map"
_FIND = (
    "https://gisprodportal.asu.edu/server/rest/services/CampusLayers/Campus_Map_Web_Layers"
    "/MapServer/find"
)
# Dining, library, museum, the building layers, services and parking.
_LAYERS = "80,98,99,101,102,103,104,105,106,107,108,109,112"
_MAX_PLACES = 8
_DESCRIPTION_CHARS = 240


def answer(params: dict[str, str]) -> tuple[str, str]:
    """Searches the campus map layers for the place by name and description."""
    place = text(params, "place")
    found = get_json(
        url(
            _FIND,
            [
                ("searchText", place),
                ("layers", _LAYERS),
                ("searchFields", "Name,Description"),
                ("returnGeometry", "true"),
                ("sr", "4326"),
                ("f", "json"),
            ],
        )
    )
    if "error" in found:
        raise QueryError(f"the campus map refused the search: {found['error']}")
    return MAP, render(place, found.get("results", []))


def render(place: str, results: list[dict[str, Any]]) -> str:
    """One line per distinct place: name, type, what it is, and a map link."""
    lines: list[str] = []
    seen: set[str] = set()
    for result in results:
        attributes = result.get("attributes", {})
        name = plain(attributes.get("Name"), 100)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        geometry = result.get("geometry") or {}
        where = ""
        if "x" in geometry and "y" in geometry:
            where = (
                " Map: https://www.google.com/maps/search/?api=1"
                f"&query={geometry['y']:.6f},{geometry['x']:.6f}"
            )
        kind = plain(attributes.get("Type"), 80)
        about = plain(attributes.get("Description"), _DESCRIPTION_CHARS)
        lines.append(f"{name} ({kind}): {about}{where}")
        if len(lines) == _MAX_PLACES:
            break
    if not lines:
        return f"No place on the ASU campus map matches {place!r}."
    return "\n".join(lines)


QUERY = QuerySource(
    key="campus_map",
    description="Find a building or place on the ASU campus map, with a map link.",
    params=(QueryParam("place", "Building name, code or place.", required=True, example="Hayden"),),
    answer=answer,
    category="campus",
)
