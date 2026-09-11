"""Library hours: the weekly table flattened to one line per location."""

from __future__ import annotations

from scraper.core.types import Fetched, Source
from scraper.ingest.extract import is_divider, page_text, plain, table_cells


def extract_hours(fetched: Fetched) -> str:
    """One line per library, carrying every day column with its hours."""
    out: list[str] = []
    headers: list[str] = []
    for raw in page_text(fetched).splitlines():
        cells = table_cells(raw)
        if cells is None:
            headers = []
            line = plain(raw)
            if line:
                out.append(line)
            continue
        if is_divider(cells):
            continue
        cells = [plain(cell) for cell in cells]
        if not any(cells):
            continue
        if not headers:
            headers = cells
            continue
        line = _row_line(headers, cells)
        if line:
            out.append(line)
    return "\n".join(out).strip()


def _row_line(headers: list[str], cells: list[str]) -> str:
    label = cells[0]
    pairs = [
        f"{header} {value}".strip()
        for header, value in zip(headers[1:], cells[1:], strict=False)
        if value
    ]
    if not pairs:
        return label
    body = "; ".join(pairs)
    return f"{label}: {body}" if label else body


SOURCE = Source(
    key="library_hours",
    url="https://lib.asu.edu/hours",
    category="library",
    fetch_every_hours=6,
    needs_js=True,
    extractor=extract_hours,
)
