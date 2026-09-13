"""HTML to normalized text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from bs4 import BeautifulSoup

from scraper.core.types import Fetched

_DROP_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe")
_BLOCK_TAGS = (
    "p",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "br",
    "div",
    "section",
    "article",
)
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_MARKER = re.compile(r"^\s*(?:[#>]+|[-*+]|\d+\.)\s*")
_DIVIDER_CELL = re.compile(r"^:?-{2,}:?$")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s")
_LINK_ONLY = re.compile(r"^\s*[-*+]?\s*\[[^\]]+\]\([^)]*\)\s*$")
_LABEL = re.compile(
    r"^(?:\*\*)?(?P<label>[A-Za-z][A-Za-z &/'-]{1,28}?)(?:\*\*)?\s*:\s*(?P<value>\S.*)$"
)
_FIELDS = " | "


def extract_text(html: bytes | str, *, main_only: bool = True, keep_forms: bool = False) -> str:
    """Visible text with one block per line, navigation and boilerplate removed. keep_forms
    keeps the text of forms, for pages that render their results inside one."""
    soup = BeautifulSoup(html, "lxml")
    dropped = tuple(t for t in _DROP_TAGS if not (keep_forms and t == "form"))
    for tag in soup(dropped):
        tag.decompose()
    root = soup
    if main_only:
        main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article")
        if main is not None:
            root = main
    for tag in root.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = root.get_text(" ")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return _BLANKS.sub("\n\n", text).strip()


def title_of(html: bytes | str) -> str | None:
    """The page title, else the first h1, else None."""
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else None


def page_text(fetched: Fetched) -> str:
    """The markdown a fetcher produced, or text pulled out of the raw body."""
    return fetched.text if fetched.text is not None else extract_text(fetched.body)


def form_page_text(fetched: Fetched) -> str:
    """page_text for a page that renders its results inside a form."""
    if fetched.text is not None:
        return fetched.text
    return extract_text(fetched.body, keep_forms=True)


def plain(line: str) -> str:
    """One markdown line as readable text: breaks become spaces, images dropped, links
    reduced to their label."""
    text = _BREAK.sub(" ", line)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = text.replace("**", "").replace("`", "")
    text = _MARKER.sub("", text)
    return _WS.sub(" ", text).strip()


def table_cells(line: str) -> list[str] | None:
    """Cells of a markdown table row, or None when the line is not one.

    A row runs from its opening pipe to its closing pipe. Text after the closing pipe is
    dropped.
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    close = stripped.rfind("|")
    if close == 0:
        return None
    return [cell.strip() for cell in stripped[1:close].split("|")]


def is_divider(cells: list[str]) -> bool:
    """True for the dashed row markdown puts under a table header."""
    return bool(cells) and all(_DIVIDER_CELL.match(cell.replace(" ", "")) for cell in cells)


def starts_record(raw: str) -> bool:
    """True when a markdown line names a new listing: a heading, or a line that is only a link."""
    return bool(_HEADING_LINE.match(raw) or _LINK_ONLY.match(raw))


def labeled(line: str) -> tuple[str, str] | None:
    """The Label: value split of a line that opens with a short label, or None."""
    match = _LABEL.match(line.strip())
    if match is None:
        return None
    return match["label"].strip(), match["value"].strip()


def blocks(text: str) -> list[tuple[str | None, list[str]]]:
    """Lines grouped under the line that names each listing; the leading group is named None."""
    groups: list[tuple[str | None, list[str]]] = [(None, [])]
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if starts_record(raw):
            groups.append((plain(raw), []))
            continue
        groups[-1][1].append(raw)
    return [(name, lines) for name, lines in groups if name or lines]


def flatten_tables(lines: Iterable[str]) -> list[str]:
    """The same lines with every markdown table row rewritten as first cell plus Header value."""
    out: list[str] = []
    headers: list[str] = []
    for raw in lines:
        cells = table_cells(raw)
        if cells is None:
            headers = []
            out.append(raw)
            continue
        if is_divider(cells):
            continue
        cells = [plain(cell) for cell in cells]
        if not any(cells):
            continue
        if not headers:
            headers = cells
            continue
        row = _row_fields(headers, cells)
        if row:
            out.append(row)
    return out


def listing_lines(
    name: str, lines: Iterable[str], labels: Mapping[str, Sequence[str]]
) -> list[str]:
    """A listing as its name plus the first value found for each label, then the lines left over."""
    found: dict[str, str] = {}
    rest: list[str] = []
    for line in lines:
        pair = None if _FIELDS in line else labeled(line)
        key = _label_key(pair[0], labels) if pair else None
        if pair is None or key is None or key in found:
            rest.append(line)
            continue
        found[key] = pair[1]
    parts = [name, *(f"{key} {found[key]}" for key in labels if key in found)]
    return [_FIELDS.join(part for part in parts if part), *rest]


def _label_key(label: str, labels: Mapping[str, Sequence[str]]) -> str | None:
    low = label.lower()
    for key, words in labels.items():
        if any(low.startswith(word) for word in words):
            return key
    return None


def _row_fields(headers: list[str], cells: list[str]) -> str:
    parts = [cells[0]] if cells and cells[0] else []
    parts += [
        f"{header} {value}".strip()
        for header, value in zip(headers[1:], cells[1:], strict=False)
        if value
    ]
    return _FIELDS.join(parts)
