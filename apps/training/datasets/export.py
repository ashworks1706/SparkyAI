"""Pull `llm` spans from Phoenix and turn each into a `TrainingExample`.

Only the engine records full prompts (as JSON on the `llm` span), so Phoenix is the source.
The JSONL trace holds events, not prompts, and is used by evals rather than here.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from training.core.settings import settings
from training.core.types import Message, TrainingExample

_QUERY = """
{ projects { edges { node { name spans(first: 1000) { edges { node {
  name attributes context { spanId traceId } } } } } } } }
"""


class ExportError(RuntimeError):
    pass


def span_to_example(span: dict[str, Any]) -> TrainingExample | None:
    """One Phoenix span row → example, or None when it is not a complete `llm` span."""
    if span.get("name") != "llm":
        return None
    attrs = json.loads(span.get("attributes") or "{}")
    inp = (attrs.get("input") or {}).get("value")
    out = (attrs.get("output") or {}).get("value")
    if not inp or not out:
        return None
    try:
        messages = [Message.model_validate(m) for m in json.loads(inp)]
        response = Message.model_validate(json.loads(out))
    except (json.JSONDecodeError, ValueError) as e:
        raise ExportError(f"span {span.get('context', {}).get('spanId')}: {e}") from e
    llm = attrs.get("llm") or {}
    params = llm.get("invocation_parameters")
    tool_count = 0
    if isinstance(params, str):
        try:
            tool_count = int(json.loads(params).get("tools", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            tool_count = 0
    return TrainingExample(
        id=span["context"]["spanId"],
        messages=messages,
        response=response,
        model=llm.get("model_name"),
        session_id=(attrs.get("session") or {}).get("id"),
        user_id=(attrs.get("user") or {}).get("id"),
        tool_count=tool_count,
    )


def fetch_spans(phoenix_url: str | None = None) -> list[dict[str, Any]]:
    url = (phoenix_url or settings().training.phoenix_url).rstrip("/") + "/graphql"
    r = httpx.post(url, json={"query": _QUERY}, timeout=60.0)
    r.raise_for_status()
    body = r.json()
    if "errors" in body:
        raise ExportError(str(body["errors"])[:300])
    return [
        e["node"] for p in body["data"]["projects"]["edges"] for e in p["node"]["spans"]["edges"]
    ]


def export_examples(phoenix_url: str | None = None) -> list[TrainingExample]:
    examples = []
    for span in fetch_spans(phoenix_url):
        ex = span_to_example(span)
        if ex is not None:
            examples.append(ex)
    return examples
