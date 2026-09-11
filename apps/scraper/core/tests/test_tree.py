import json

import httpx
import numpy as np
import pytest
from scraper.core.settings import Scraper
from scraper.core.types import SummaryError
from scraper.tree import (
    MIN_CLUSTERS,
    TreeParams,
    build_tree,
    cluster,
    cluster_count,
    summarize_cluster,
)

DIM = 1024
DEFAULTS = TreeParams.from_settings(Scraper())


def vector(seed: int, center: int) -> list[float]:
    """A 1024-dim point near the axis named by center."""
    rng = np.random.default_rng(seed)
    v = rng.normal(scale=0.01, size=DIM)
    v[center] += 1.0
    return [float(x) for x in v]


def leaves(count: int, centers: int = 4) -> tuple[list[str], list[list[float]]]:
    texts = [f"leaf {i} about topic {i % centers}" for i in range(count)]
    vectors = [vector(i, i % centers) for i in range(count)]
    return texts, vectors


def fake_summarize(texts):
    return "summary of " + " | ".join(texts)


def fake_embed(texts):
    return [vector(1_000 + i, i % 4) for i, _ in enumerate(texts)]


def params(**over) -> TreeParams:
    fields = {
        "max_level": DEFAULTS.max_level,
        "cluster_size": DEFAULTS.cluster_size,
        "min_chunks": DEFAULTS.min_chunks,
    }
    fields.update(over)
    return TreeParams(**fields)


def chat_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://chat/v1")


def test_cluster_count_covers_every_row() -> None:
    assert cluster_count(20, 5) == 4
    assert cluster_count(21, 5) == 5
    assert cluster_count(1, 5) == 1


def test_clustering_groups_points_that_share_a_center() -> None:
    vectors = [vector(i, 0) for i in range(6)] + [vector(10 + i, 7) for i in range(6)]
    groups = cluster(vectors, 2)
    assert sorted(len(g) for g in groups) == [6, 6]
    assert {frozenset(g) for g in groups} == {frozenset(range(6)), frozenset(range(6, 12))}


def test_a_source_with_too_few_chunks_builds_no_tree() -> None:
    texts, vectors = leaves(DEFAULTS.min_chunks - 1)
    assert build_tree(texts, vectors, params=params(), summarize=_never, embed=_never) == []


def test_parents_summarize_the_leaves_they_cover() -> None:
    texts, vectors = leaves(20)
    nodes = build_tree(texts, vectors, params=params(), summarize=fake_summarize, embed=fake_embed)
    assert nodes
    first = nodes[0]
    assert first.level == 1
    assert first.ordinal == len(texts)
    assert len(first.embedding) == DIM
    # A parent's content is what the model returned for exactly the rows it covers.
    assert first.content == fake_summarize([texts[i] for i in first.children])
    # Ordinals continue past the leaves, so nothing collides on (version_id, ordinal).
    assert [n.ordinal for n in nodes] == list(range(len(texts), len(texts) + len(nodes)))


def test_every_child_position_points_below_its_parent() -> None:
    texts, vectors = leaves(60)
    nodes = build_tree(texts, vectors, params=params(), summarize=fake_summarize, embed=fake_embed)
    for position, node in enumerate(nodes):
        for child in node.children:
            assert child < len(texts) + position
    covered = [child for node in nodes for child in node.children]
    assert len(covered) == len(set(covered))
    # Level 1 covers every leaf exactly once.
    assert sorted(c for n in nodes if n.level == 1 for c in n.children) == list(range(len(texts)))


def test_the_level_ceiling_stops_the_recursion() -> None:
    texts, vectors = leaves(200)
    nodes = build_tree(
        texts, vectors, params=params(max_level=1), summarize=fake_summarize, embed=fake_embed
    )
    assert nodes
    assert {n.level for n in nodes} == {1}


def test_the_cluster_floor_stops_the_recursion() -> None:
    texts, vectors = leaves(30)
    nodes = build_tree(
        texts, vectors, params=params(max_level=9), summarize=fake_summarize, embed=fake_embed
    )
    top = max(n.level for n in nodes)
    assert top < 9
    # Recursion stopped with a level the floor would not let it summarize again.
    assert cluster_count(len([n for n in nodes if n.level == top]), DEFAULTS.cluster_size) < (
        MIN_CLUSTERS
    )


def test_a_level_that_would_not_shrink_is_not_built() -> None:
    texts, vectors = leaves(20)
    assert (
        build_tree(texts, vectors, params=params(cluster_size=1), summarize=_never, embed=_never)
        == []
    )


def test_a_summary_is_read_from_the_chat_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "[1] one" in body["messages"][-1]["content"]
        assert "[2] two" in body["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": " a summary "}}]})

    assert summarize_cluster(["one", "two"], client=chat_client(handler)) == "a summary"


def test_an_empty_summary_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    with pytest.raises(SummaryError, match="empty summary"):
        summarize_cluster(["one", "two"], client=chat_client(handler))


def test_a_refused_chat_call_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="no slot")

    with pytest.raises(SummaryError, match="503"):
        summarize_cluster(["one"], client=chat_client(handler))


def test_an_unreadable_chat_body_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(SummaryError, match="unreadable"):
        summarize_cluster(["one"], client=chat_client(handler))


def _never(*_args, **_kwargs):
    raise AssertionError("no model call is made when no tree is built")
