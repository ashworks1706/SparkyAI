"""fetch → hash → snapshot → extract → chunk → embed → index. One run per source."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from scraper import chunk, embed, extract, fetch
from scraper.settings import settings
from scraper.sources import Source
from scraper.store import object as objects
from scraper.store import postgres

log = structlog.get_logger()


@dataclass(frozen=True)
class RunResult:
    source: str
    changed: bool
    chunks: int
    content_hash: str


def run_source(source: Source, *, force: bool = False) -> RunResult:
    """Ingests one source. Skips everything after the fetch when the page hash is unchanged."""
    cfg = settings()
    fetched = fetch.fetch(source.url, needs_js=source.needs_js)
    content_hash = hashlib.sha256(fetched.body).hexdigest()
    fetched_at = datetime.now(UTC)

    with postgres.connection() as conn:
        row = postgres.upsert_source(
            conn, source.key, source.url, source.category, source.fetch_every_hours
        )
        previous = postgres.latest_version(conn, row.id)
        if previous and previous["content_hash"] == content_hash and not force:
            log.info("unchanged", source=source.key)
            conn.commit()
            return RunResult(source.key, changed=False, chunks=0, content_hash=content_hash)

        snapshot_key = f"{source.key}/{fetched_at:%Y%m%dT%H%M%SZ}-{content_hash[:12]}.html"
        objects.put_snapshot(snapshot_key, fetched.body, fetched.content_type)

        text = extract.extract_text(fetched.body)
        title = extract.title_of(fetched.body) or source.key
        pieces = chunk.chunk_text(
            text,
            max_chars=cfg.scraper.chunk_chars,
            overlap_chars=cfg.scraper.chunk_overlap_chars,
        )
        # Prefix each chunk with the page title so the embedding carries the source context.
        texts = [f"{title}\n{p}" for p in pieces]
        vectors = embed.embed_texts(texts)

        version_id = postgres.insert_version(
            conn,
            source_id=row.id,
            content_hash=content_hash,
            snapshot_key=snapshot_key,
            parser_version=cfg.scraper.parser_version,
            chunker_version=cfg.scraper.chunker_version,
            embedding_model=cfg.embedding.name,
            previous_id=previous["id"] if previous else None,
        )
        written = postgres.replace_chunks(
            conn,
            tenant_id=cfg.scraper.tenant_id,
            source=row,
            version_id=version_id,
            fetched_at=fetched_at,
            chunks=[
                postgres.ChunkRow(i, t, v)
                for i, (t, v) in enumerate(zip(texts, vectors, strict=True))
            ],
        )
        conn.commit()
    log.info("indexed", source=source.key, chunks=written, hash=content_hash[:12])
    return RunResult(source.key, changed=True, chunks=written, content_hash=content_hash)
