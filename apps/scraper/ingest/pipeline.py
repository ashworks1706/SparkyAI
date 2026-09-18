"""Fetch, hash, snapshot, extract, chunk, embed, index. One run per source or live-query page."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import psycopg
import structlog

from scraper.core import telemetry
from scraper.core.settings import settings
from scraper.core.types import ChunkRow, Fetched, PipelineError, RunResult, Source, SourceRow
from scraper.ingest import chunk, embed, extract, fetch, tree
from scraper.ingest.pace import HostPacer
from scraper.store import object as objects
from scraper.store import postgres

log = structlog.get_logger()


def run_source(source: Source, *, force: bool = False, pacer: HostPacer | None = None) -> RunResult:
    """Ingests one source. Skips everything after the fetch when the page hash is unchanged."""
    with telemetry.tracer().start_as_current_span(
        "scrape.source",
        attributes={
            "sparky.input": source.url,
            "sparky.source": source.key,
        },
    ) as span:
        result = _run_source(source, force=force, pacer=pacer)
        span.set_attribute("sparky.output", _outcome(result))
        return result


def index_page(source: Source, fetched: Fetched, *, force: bool = False) -> RunResult:
    """Indexes a page that was already fetched, under source, the way a scheduled run would."""
    with telemetry.tracer().start_as_current_span(
        "scrape.index",
        attributes={
            "sparky.input": fetched.url,
            "sparky.source": source.key,
        },
    ) as span:
        result = _index(source, _register(source), fetched, force=force)
        span.set_attribute("sparky.output", _outcome(result))
        return result


def check_quality_floor(
    key: str, text_chars: int, previous_chars: int | None, *, ratio: float, min_chars: int
) -> None:
    """Raises when this run extracted materially less than the last indexed version did."""
    if previous_chars is None or previous_chars < min_chars:
        return
    floor = int(previous_chars * ratio)
    if text_chars < floor:
        raise PipelineError(
            f"{key}: extraction produced {text_chars} chars against {previous_chars} in the "
            f"last indexed version, under the floor of {floor}; refusing to replace the index. "
            "Re-run with --force to accept it"
        )


def _run_source(source: Source, *, force: bool, pacer: HostPacer | None) -> RunResult:
    # The attempt is committed before the fetch. A failed or unchanged run still counts as one.
    row = _register(source)
    if pacer is not None:
        pacer.wait(source.url)
    fetched = fetch.fetch(source.url, needs_js=source.needs_js)
    return _index(source, row, fetched, force=force)


def _register(source: Source) -> SourceRow:
    """Upserts the sources row, stamps the attempt, and commits."""
    with postgres.connection() as conn:
        row = postgres.upsert_source(
            conn, source.key, source.url, source.category, source.fetch_every_hours
        )
        conn.commit()
    return row


def _outcome(result: RunResult) -> str:
    return f"{'indexed' if result.changed else 'unchanged'}: {result.chunks} chunks"


def _index(source: Source, row: SourceRow, fetched: Fetched, *, force: bool) -> RunResult:
    """Everything after the fetch: hash, snapshot, extract, chunk, embed, write, tree."""
    cfg = settings()
    content_hash = hashlib.sha256(fetched.body).hexdigest()
    fetched_at = datetime.now(UTC)

    with postgres.connection() as conn:
        previous = postgres.latest_version(conn, row.id)
        if previous and previous["content_hash"] == content_hash and not force:
            log.info("unchanged", source=source.key)
            conn.commit()
            return RunResult(source.key, changed=False, chunks=0, content_hash=content_hash)

        ext = "md" if fetched.content_type.startswith("text/markdown") else "html"
        snapshot_key = f"{source.key}/{fetched_at:%Y%m%dT%H%M%SZ}-{content_hash[:12]}.{ext}"
        objects.put_snapshot(snapshot_key, fetched.body, fetched.content_type)

        if source.extractor is not None:
            text = source.extractor(fetched)
            html_title = None if fetched.text is not None else extract.title_of(fetched.body)
            title = fetched.title or html_title or source.key
        elif fetched.text is not None:
            text = fetched.text
            title = fetched.title or source.key
        else:
            text = extract.extract_text(fetched.body)
            title = extract.title_of(fetched.body) or source.key
        pieces = chunk.chunk_text(
            text,
            max_chars=cfg.scraper.chunk_chars,
            overlap_chars=cfg.scraper.chunk_overlap_chars,
        )
        if not pieces:
            # replace_chunks deletes before it inserts.
            raise PipelineError(
                f"{source.key}: extraction produced no text from {len(fetched.body)} bytes; "
                "refusing to replace the index with nothing"
            )
        if not force:
            check_quality_floor(
                source.key,
                len(text),
                previous["text_chars"] if previous else None,
                ratio=cfg.scraper.quality_floor_ratio,
                min_chars=cfg.scraper.quality_floor_min_chars,
            )
        postgres.set_source_title(conn, row.id, title)
        # Each embedded text is prefixed with the page title.
        texts = [f"{title}\n{p}" for p in pieces]
        vectors = embed.embed_texts(texts)

        version_id = postgres.insert_version(
            conn,
            source_id=row.id,
            content_hash=content_hash,
            snapshot_key=snapshot_key,
            parser_version=cfg.scraper.parser_version,
            chunker_version=cfg.scraper.chunker_version(),
            embedding_model=cfg.embedding.name,
            previous_id=previous["id"] if previous else None,
            text_chars=len(text),
            chunk_count=len(pieces),
        )
        written = postgres.replace_chunks(
            conn,
            tenant_id=cfg.scraper.tenant_id,
            source=row,
            version_id=version_id,
            fetched_at=fetched_at,
            chunks=[ChunkRow(i, t, v) for i, (t, v) in enumerate(zip(texts, vectors, strict=True))],
        )
        _build_tree(
            conn,
            key=source.key,
            row=row,
            version_id=version_id,
            fetched_at=fetched_at,
            texts=texts,
            vectors=vectors,
        )
        conn.commit()
    log.info("indexed", source=source.key, chunks=written, hash=content_hash[:12])
    return RunResult(source.key, changed=True, chunks=written, content_hash=content_hash)


def _build_tree(
    conn: psycopg.Connection,
    *,
    key: str,
    row: SourceRow,
    version_id: uuid.UUID,
    fetched_at: datetime,
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Rebuilds this source's summary levels over the leaves replace_chunks just wrote."""
    cfg = settings()
    if not cfg.scraper.tree_enabled:
        return
    nodes = tree.build_tree(texts, vectors, params=tree.TreeParams.from_settings(cfg.scraper))
    if not nodes:
        return
    postgres.insert_tree(
        conn,
        tenant_id=cfg.scraper.tenant_id,
        source=row,
        version_id=version_id,
        fetched_at=fetched_at,
        leaves=postgres.leaf_ids(conn, version_id),
        nodes=nodes,
    )
    log.info("tree", source=key, nodes=len(nodes), levels=nodes[-1].level)
