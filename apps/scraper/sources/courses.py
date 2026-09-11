"""Courses: each catalog entry kept as one line carrying its number, title and units."""

from __future__ import annotations

import re

from scraper.core.types import Fetched, Source
from scraper.extract import is_divider, page_text, plain, table_cells

_NUMBER = re.compile(r"^[A-Z]{2,4}\s?\d{3}[A-Z]?$")
_IN_LINE = re.compile(
    r"^(?P<number>[A-Z]{2,4}\s?\d{3}[A-Z]?)\s*[:.\-]?\s+"
    r"(?P<title>.+?)\s*(?:\((?P<units>\d+(?:\.\d+)?)(?:\s*credit[s]?)?\)\s*)?$"
)
_UNITS = re.compile(r"^\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?$")


def extract_courses(fetched: Fetched) -> str:
    """One line per course: number, title, units. Catalog prose is kept as it reads."""
    out: list[str] = []
    for raw in page_text(fetched).splitlines():
        cells = table_cells(raw)
        if cells is not None:
            if is_divider(cells):
                continue
            line = _from_cells([plain(cell) for cell in cells])
        else:
            line = _from_line(plain(raw))
        if line:
            out.append(line)
    return "\n".join(out).strip()


def _from_cells(cells: list[str]) -> str:
    filled = [cell for cell in cells if cell]
    index = next((i for i, cell in enumerate(filled) if _NUMBER.match(cell)), None)
    if index is None:
        return " | ".join(filled)
    number = filled[index]
    rest = filled[:index] + filled[index + 1 :]
    units_index = next((i for i, cell in enumerate(rest) if _UNITS.match(cell)), None)
    units = rest.pop(units_index) if units_index is not None else None
    return _course_line(number, max(rest, key=len, default=""), units)


def _from_line(line: str) -> str:
    match = _IN_LINE.match(line)
    if match is None:
        return line
    return _course_line(match["number"], match["title"], match["units"])


def _course_line(number: str, title: str, units: str | None) -> str:
    parts = [number, title] if title else [number]
    if units:
        parts.append(f"{units} unit" if units == "1" else f"{units} units")
    return " | ".join(parts)


SOURCE = Source(
    key="courses",
    url="https://catalog.asu.edu/",
    category="courses",
    fetch_every_hours=168,
    needs_js=False,
    extractor=extract_courses,
)
