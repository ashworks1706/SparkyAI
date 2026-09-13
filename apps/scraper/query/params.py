"""Turning the parameters a model passes into the values a search URL takes.

Values reach these helpers after registry.check, so every choice is already one of the keys of
its mapping.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable

from scraper.core.types import QueryError

# ASU term codes are 2 plus the two-digit calendar year plus the session digit.
# Fall 2026 is 2267.
_TERM_DIGIT = {"spring": "1", "summer": "4", "fall": "7"}


def term_code(term: str) -> str:
    """Fall 2026 to 2267. Raises QueryError on anything else."""
    parts = term.strip().split()
    if len(parts) != 2:
        raise QueryError(f"term must look like 'Fall 2026', got {term!r}")
    season, year = parts[0].lower(), parts[1]
    if season not in _TERM_DIGIT:
        raise QueryError(f"term season must be spring, summer or fall, got {parts[0]!r}")
    if not (year.isdigit() and len(year) == 4):
        raise QueryError(f"term year must be four digits, got {year!r}")
    return f"2{year[2:]}{_TERM_DIGIT[season]}"


def text(params: dict[str, str], name: str) -> str:
    """The value of name with surrounding whitespace removed, or empty."""
    return params.get(name, "").strip()


def codes(params: dict[str, str], name: str, mapping: dict[str, str]) -> list[str]:
    """The codes of a comma-separated list of choices."""
    return [mapping[v.strip().lower()] for v in text(params, name).split(",") if v.strip()]


def code(params: dict[str, str], name: str, mapping: dict[str, str]) -> str:
    """The code of a single choice, or empty when the parameter is absent."""
    value = text(params, name).lower()
    return mapping[value] if value else ""


def choices(mapping: dict[str, str]) -> tuple[str, ...]:
    """The choices a mapping accepts, in order."""
    return tuple(mapping)


def url(base: str, pairs: Iterable[tuple[str, str]]) -> str:
    """base with every non-empty pair as a query parameter, in order, repeats kept."""
    kept = [(k, v) for k, v in pairs if v]
    return f"{base}?{urllib.parse.urlencode(kept)}" if kept else base
