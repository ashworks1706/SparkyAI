"""Embedding client against the llama-server embed endpoint (OpenAI-compatible)."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from scraper.core.settings import settings
from scraper.core.types import EmbedError


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """One vector per text, same order. Batched; dimension checked against settings."""
    cfg = settings().embedding
    if not texts:
        return []
    headers = {}
    key = cfg.api_key.get_secret_value()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    out: list[list[float]] = []
    with httpx.Client(base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=120.0) as http:
        for start in range(0, len(texts), cfg.batch_size):
            batch = list(texts[start : start + cfg.batch_size])
            r = http.post("/embeddings", json={"model": cfg.name, "input": batch})
            if r.status_code >= 400:
                raise EmbedError(f"embed endpoint returned {r.status_code}: {r.text[:300]}")
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            vectors = [d["embedding"] for d in data]
            if len(vectors) != len(batch):
                raise EmbedError(f"asked for {len(batch)} vectors, got {len(vectors)}")
            for v in vectors:
                if len(v) != cfg.dim:
                    raise EmbedError(f"dimension {len(v)} does not match configured {cfg.dim}")
            out.extend(vectors)
    return out
