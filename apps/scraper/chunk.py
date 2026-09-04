"""Paragraph-aware chunking with overlap."""

from __future__ import annotations


def chunk_text(text: str, *, max_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """Splits on paragraph boundaries, packing paragraphs up to `max_chars`. Paragraphs longer
    than the limit are split on sentence-ish boundaries. Consecutive chunks share `overlap_chars`
    of trailing context so facts straddling a boundary survive."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current}\n{piece}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n{piece}".strip() if tail else piece
            if len(current) > max_chars:
                chunks.append(current[:max_chars])
                current = current[max_chars - overlap_chars :] if overlap_chars else ""
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out: list[str] = []
    start = 0
    while start < len(paragraph):
        end = min(start + max_chars, len(paragraph))
        if end < len(paragraph):
            cut = max(
                paragraph.rfind(". ", start, end),
                paragraph.rfind("; ", start, end),
                paragraph.rfind(", ", start, end),
            )
            if cut > start + max_chars // 2:
                end = cut + 1
        out.append(paragraph[start:end].strip())
        start = end
    return [o for o in out if o]
