"""Checks model-supplied parameters against a source and maps them to search URL values."""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable

from scraper.core.types import QueryError, QuerySource

# ASU term codes are 2 plus the two-digit calendar year plus the session digit; Fall 2026 is 2267.
_TERM_DIGIT = {"spring": "1", "summer": "4", "fall": "7"}


def check(source: QuerySource, params: dict[str, str]) -> None:
    """Raises QueryError when a required parameter is missing, unknown, or not among its choices."""
    missing = [p.name for p in source.params if p.required and not params.get(p.name, "").strip()]
    if missing:
        raise QueryError(f"{source.key} needs: {', '.join(missing)}")
    unknown = set(params) - {p.name for p in source.params}
    if unknown:
        taken = ", ".join(p.name for p in source.params) or "nothing"
        raise QueryError(
            f"{source.key} has no parameter {', '.join(sorted(unknown))}; it takes: {taken}"
        )
    for p in source.params:
        value = params.get(p.name, "").strip()
        if not p.choices or not value:
            continue
        values = [v.strip() for v in value.split(",") if v.strip()] if p.many else [value]
        if not p.many and "," in value:
            raise QueryError(f"{p.name} takes one value, got {value!r}")
        allowed = {c.lower() for c in p.choices}
        for v in values:
            if v.lower() not in allowed:
                raise QueryError(f"{p.name} {v!r} is not one of: {', '.join(p.choices)}")


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
