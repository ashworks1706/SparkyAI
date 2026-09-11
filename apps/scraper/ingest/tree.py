"""The hierarchical index over one source: cluster a level, summarize each cluster, embed the
summary, repeat."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx
import numpy as np
import structlog
from sklearn.cluster import KMeans

from scraper.core.settings import Scraper, Summary, settings
from scraper.core.types import PipelineError, SummaryError, TreeNode
from scraper.ingest import embed as embedding

log = structlog.get_logger()

# A level is built only when the level below splits into at least this many clusters. One
# cluster would summarize a level into a copy of itself.
MIN_CLUSTERS = 2

# k-means is seeded, so the same leaves cluster the same way on every run.
RANDOM_STATE = 0
N_INIT = 10

SYSTEM_PROMPT = (
    "You summarize passages from an Arizona State University page for a retrieval index. "
    "Write one dense paragraph that keeps the facts a student would search for: names, "
    "dates, deadlines, amounts, locations, requirements, and steps. Use only what the "
    "passages say. Do not add a preamble, a heading, or a closing line."
)

#: Turns the texts of one cluster into the text of its parent.
Summarize = Callable[[Sequence[str]], str]

#: Turns texts into one vector each, in order.
Embed = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class TreeParams:
    """What bounds the recursion and how wide one summary reaches."""

    max_level: int
    cluster_size: int
    min_chunks: int

    @staticmethod
    def from_settings(cfg: Scraper) -> TreeParams:
        """The parameters the configured scraper builds trees with."""
        return TreeParams(
            max_level=cfg.tree_max_level,
            cluster_size=cfg.tree_cluster_size,
            min_chunks=cfg.tree_min_chunks,
        )


def cluster_count(rows: int, cluster_size: int) -> int:
    """How many clusters a level of this many rows is cut into."""
    if cluster_size < 1:
        raise PipelineError(f"tree cluster size must be at least 1, got {cluster_size}")
    return math.ceil(rows / cluster_size)


def cluster(vectors: Sequence[Sequence[float]], n_clusters: int) -> list[list[int]]:
    """Positions in vectors grouped by k-means label. A label no vector took is dropped."""
    matrix = np.asarray(vectors, dtype=np.float64)
    labels = KMeans(n_clusters=n_clusters, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(
        matrix
    )
    groups: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        groups.setdefault(int(label), []).append(position)
    return [groups[label] for label in sorted(groups)]


def build_tree(
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    *,
    params: TreeParams,
    summarize: Summarize | None = None,
    embed: Embed | None = None,
) -> list[TreeNode]:
    """The summary levels above one source's leaves, parents after the nodes they cover.

    Empty when the source has fewer leaves than the floor, or when the first level would not
    split the leaves into at least MIN_CLUSTERS clusters.
    """
    if len(texts) != len(vectors):
        raise PipelineError(f"{len(texts)} leaf texts against {len(vectors)} leaf vectors")
    summarize = summarize or summarize_cluster
    embed = embed or embedding.embed_texts
    leaves = len(texts)
    if leaves < params.min_chunks:
        return []

    nodes: list[TreeNode] = []
    # Position in the leaves-then-nodes list of every row in the level being clustered.
    below = list(range(leaves))
    level_texts = list(texts)
    level_vectors = [list(v) for v in vectors]

    for level in range(1, params.max_level + 1):
        n_clusters = cluster_count(len(level_texts), params.cluster_size)
        if n_clusters < MIN_CLUSTERS or n_clusters >= len(level_texts):
            break
        groups = cluster(level_vectors, n_clusters)
        if len(groups) < MIN_CLUSTERS:
            break
        summaries = [summarize([level_texts[position] for position in group]) for group in groups]
        level_vectors = [list(v) for v in embed(summaries)]
        if len(level_vectors) != len(summaries):
            raise PipelineError(
                f"level {level}: {len(summaries)} summaries against {len(level_vectors)} vectors"
            )
        above: list[int] = []
        for group, summary, vector in zip(groups, summaries, level_vectors, strict=True):
            nodes.append(
                TreeNode(
                    level=level,
                    ordinal=leaves + len(nodes),
                    content=summary,
                    embedding=vector,
                    children=tuple(below[position] for position in group),
                )
            )
            above.append(leaves + len(nodes) - 1)
        level_texts = summaries
        below = above
        log.debug("tree level", level=level, nodes=len(groups))

    return nodes


def summarize_cluster(texts: Sequence[str], *, client: httpx.Client | None = None) -> str:
    """One summary of one cluster from the chat endpoint. An empty answer is an error."""
    cfg = settings().summary
    if not texts:
        raise SummaryError("a cluster with no passages has nothing to summarize")
    if client is not None:
        return _summarize(client, cfg, texts)
    headers = {}
    key = cfg.api_key.get_secret_value()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(
        base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=cfg.timeout_secs
    ) as http:
        return _summarize(http, cfg, texts)


def _summarize(http: httpx.Client, cfg: Summary, texts: Sequence[str]) -> str:
    passages = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
    r = http.post(
        "/chat/completions",
        json={
            "model": cfg.name,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "chat_template_kwargs": {"enable_thinking": cfg.thinking},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": passages},
            ],
        },
    )
    if r.status_code >= 400:
        raise SummaryError(f"chat endpoint returned {r.status_code}: {r.text[:300]}")
    try:
        content = r.json()["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError, ValueError) as e:
        raise SummaryError(f"chat endpoint returned an unreadable body: {e}") from e
    summary = (content or "").strip()
    if not summary:
        raise SummaryError(f"chat endpoint returned an empty summary for {len(texts)} passages")
    return summary
