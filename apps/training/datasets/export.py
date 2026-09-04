"""Pull `llm` spans from Phoenix and turn each into a `TrainingExample`.

Only the engine records full prompts (as JSON on the `llm` span), so Phoenix is the source.
The JSONL trace holds events, not prompts, and is used by evals rather than here.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from training.core.settings import settings
from training.core.types import ExportError, Message, TrainingExample

_PAGE = 500
_QUERY = """
query($first: Int!, $after: String) {
  projects { edges { node { name
    spans(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      edges { node { name attributes context { spanId traceId } } }
    }
  } } }
}
"""


def span_to_example(span: dict[str, Any]) -> TrainingExample | None:
    """One Phoenix span row → example. Spans that are not `llm`, or that ended before a reply
    was recorded, are skipped; a malformed `llm` span is an error, since the engine wrote it."""
    if span.get("name") != "llm":
        return None
    span_id = span["context"]["spanId"]
    attrs = json.loads(span["attributes"])
    inp = (attrs.get("input") or {}).get("value")
    out = (attrs.get("output") or {}).get("value")
    if not inp or not out:
        return None
    try:
        messages = [Message.model_validate(m) for m in json.loads(inp)]
        response = Message.model_validate(json.loads(out))
    except (json.JSONDecodeError, ValueError) as e:
        raise ExportError(f"span {span_id}: {e}") from e
    llm = attrs.get("llm") or {}
    tool_count = 0
    params = llm.get("invocation_parameters")
    if params is not None:
        try:
            tool_count = int(json.loads(params).get("tools", 0))
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            raise ExportError(f"span {span_id}: bad invocation_parameters: {e}") from e
    return TrainingExample(
        id=span_id,
        messages=messages,
        response=response,
        model=llm.get("model_name"),
        session_id=(attrs.get("session") or {}).get("id"),
        user_id=(attrs.get("user") or {}).get("id"),
        tool_count=tool_count,
    )


def _page(url: str, after: str | None) -> dict[str, Any]:
    r = httpx.post(
        url, json={"query": _QUERY, "variables": {"first": _PAGE, "after": after}}, timeout=60.0
    )
    if r.status_code != 200:
        raise ExportError(f"phoenix returned {r.status_code}: {r.text[:200]}")
    body = r.json()
    if "errors" in body:
        raise ExportError(str(body["errors"])[:300])
    try:
        return body["data"]["projects"]
    except (KeyError, TypeError) as e:
        raise ExportError(f"unexpected graphql shape from {url}: {e}") from e


def fetch_spans(phoenix_url: str | None = None) -> list[dict[str, Any]]:
    """Every span in every project. Pages through Phoenix so nothing is silently cut off."""
    url = (phoenix_url or settings().training.phoenix_url).rstrip("/") + "/graphql"
    spans: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        projects = _page(url, after)
        more = False
        for p in projects["edges"]:
            conn = p["node"]["spans"]
            spans.extend(e["node"] for e in conn["edges"])
            if conn["pageInfo"]["hasNextPage"]:
                more = True
                after = conn["pageInfo"]["endCursor"]
        if not more:
            return spans


def export_examples(phoenix_url: str | None = None) -> list[TrainingExample]:
    examples = []
    for span in fetch_spans(phoenix_url):
        ex = span_to_example(span)
        if ex is not None:
            examples.append(ex)
    return examples
