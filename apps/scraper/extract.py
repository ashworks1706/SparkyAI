"""HTML to normalized text."""

from __future__ import annotations

import re

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
_MARKER = re.compile(r"^\s*(?:[#>]+|[-*+]|\d+\.)\s*")
_DIVIDER_CELL = re.compile(r"^:?-{2,}:?$")


def extract_text(html: bytes | str, *, main_only: bool = True) -> str:
    """Visible text with one block per line, navigation and boilerplate removed."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP_TAGS):
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
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else None


def page_text(fetched: Fetched) -> str:
    """The markdown a fetcher produced, or text pulled out of the raw body."""
    return fetched.text if fetched.text is not None else extract_text(fetched.body)


def plain(line: str) -> str:
    """One markdown line as readable text: images dropped, links reduced to their label."""
    text = _IMAGE.sub("", line)
    text = _LINK.sub(r"\1", text)
    text = text.replace("**", "").replace("`", "")
    text = _MARKER.sub("", text)
    return _WS.sub(" ", text).strip()


def table_cells(line: str) -> list[str] | None:
    """Cells of a markdown table row, or None when the line is not one."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def is_divider(cells: list[str]) -> bool:
    """True for the dashed row markdown puts under a table header."""
    return bool(cells) and all(_DIVIDER_CELL.match(cell.replace(" ", "")) for cell in cells)
