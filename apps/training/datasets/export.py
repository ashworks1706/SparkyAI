"""Pull llm spans from PostHog and turn each into a TrainingExample.

The engine records the full prompt and the reply as JSON attributes on the llm span. Spans are
read from posthog.trace_spans through the HogQL query API, paged by a (timestamp, uuid) keyset
cursor. The JSONL trace holds events, not prompts, and is read by evals.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from training.core.settings import Training, settings
from training.core.types import ExportError, Message, TrainingExample

_INPUT = "gen_ai.input.messages"
_OUTPUT = "gen_ai.output.messages"
_MODEL = "gen_ai.request.model"
_SESSION = "$ai_session_id"
_USER = "posthog.distinct_id"
_TOOLS = "sparky.tools"

_SELECT = "SELECT uuid, timestamp, attributes FROM posthog.trace_spans WHERE name = 'llm'"
_ORDER = " ORDER BY timestamp ASC, uuid ASC LIMIT {limit}"
_AFTER = (
    " AND (timestamp > toDateTime('{ts}', 'UTC')"
    " OR (timestamp = toDateTime('{ts}', 'UTC') AND uuid > '{uuid}'))"
)


def hogql(limit: int, after: tuple[str, str] | None = None) -> str:
    """The HogQL page query. after is the (timestamp, uuid) of the last row already read."""
    query = _SELECT
    if after is not None:
        ts, row_id = after
        query += _AFTER.format(ts=_cursor_timestamp(ts), uuid=_cursor_uuid(row_id))
    return query + _ORDER.format(limit=int(limit))


def _cursor_timestamp(raw: str) -> str:
    """A result timestamp as a UTC literal HogQL parses, microsecond precision."""
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ExportError(f"bad span timestamp {raw!r}: {e}") from e
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def _cursor_uuid(raw: str) -> str:
    try:
        return str(uuid.UUID(raw))
    except ValueError as e:
        raise ExportError(f"bad span uuid {raw!r}: {e}") from e


def _json_value(value: Any, what: str, row_id: str) -> Any:
    """Parses a JSON string; any other value is returned as is."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ExportError(f"span {row_id}: {what} is not JSON: {e}") from e


def _text_of(items: list[Any], key: str, row_id: str) -> str:
    """Joins text items of shape {type: text, <key>: str}. Any other item raises."""
    texts = []
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("type") != "text"
            or not isinstance(item.get(key), str)
        ):
            raise ExportError(f"span {row_id}: unsupported message part {item!r:.200}")
        texts.append(item[key])
    return "".join(texts)


def _message(raw: Any, row_id: str) -> Message:
    """One message in the engine Message shape, a parts shape, or a text content list."""
    if not isinstance(raw, dict):
        raise ExportError(f"span {row_id}: message is not an object: {raw!r:.200}")
    data = dict(raw)
    if "parts" in data:
        parts = data.pop("parts")
        if not isinstance(parts, list) or "content" in data:
            raise ExportError(f"span {row_id}: bad parts message {raw!r:.200}")
        data["content"] = _text_of(parts, "content", row_id)
    elif isinstance(data.get("content"), list):
        data["content"] = _text_of(data["content"], "text", row_id)
    elif data.get("content") is None:
        data.pop("content", None)
    try:
        return Message.model_validate(data)
    except ValueError as e:
        raise ExportError(f"span {row_id}: bad message: {e}") from e


def _messages(value: Any, what: str, row_id: str) -> list[Message]:
    parsed = _json_value(value, what, row_id)
    if not isinstance(parsed, list):
        raise ExportError(f"span {row_id}: {what} is not a list")
    return [_message(m, row_id) for m in parsed]


def _tool_count(value: Any, row_id: str) -> int:
    """sparky.tools as a count: a list of tool definitions, its JSON, or an integer."""
    if value is None:
        return 0
    parsed = _json_value(value, _TOOLS, row_id)
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return parsed
    raise ExportError(f"span {row_id}: bad {_TOOLS} {value!r:.200}")


def row_to_example(row: dict[str, Any]) -> TrainingExample | None:
    """One HogQL result row to an example. Rows that are not llm, or that ended before a reply
    was recorded, are skipped. A malformed llm row raises."""
    row_id = str(row.get("uuid"))
    attrs = _json_value(row.get("attributes"), "attributes", row_id)
    if not isinstance(attrs, dict):
        raise ExportError(f"span {row_id}: attributes is not an object")
    span = attrs.get("sparky.span")
    if span is not None and span != "llm":
        return None
    inp = attrs.get(_INPUT)
    out = attrs.get(_OUTPUT)
    if inp in (None, "", []) or out in (None, "", []):
        return None
    messages = _messages(inp, _INPUT, row_id)
    choices = _messages(out, _OUTPUT, row_id)
    if not choices:
        return None
    session = attrs.get(_SESSION)
    user = attrs.get(_USER)
    return TrainingExample(
        id=row_id,
        messages=messages,
        response=choices[0],
        model=attrs.get(_MODEL),
        session_id=str(session) if session else None,
        user_id=str(user) if user else None,
        tool_count=_tool_count(attrs.get(_TOOLS), row_id),
    )


def _page(client: httpx.Client, url: str, key: str, query: str) -> list[dict[str, Any]]:
    try:
        r = client.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"query": {"kind": "HogQLQuery", "query": query}, "name": "sparky training"},
        )
    except httpx.HTTPError as e:
        raise ExportError(f"posthog request failed: {e}") from e
    if r.status_code != 200:
        raise ExportError(f"posthog returned {r.status_code}: {r.text[:200]}")
    try:
        body = r.json()
        columns = body["columns"]
        results = body["results"]
    except (ValueError, KeyError, TypeError) as e:
        raise ExportError(f"unexpected query response shape: {e}") from e
    if not isinstance(columns, list) or not isinstance(results, list):
        raise ExportError("unexpected query response shape: columns or results is not a list")
    return [dict(zip(columns, row, strict=True)) for row in results]


def fetch_rows(
    cfg: Training | None = None, transport: httpx.BaseTransport | None = None
) -> list[dict[str, Any]]:
    """Every llm span row, paged through the PostHog HogQL query API."""
    cfg = cfg or settings().training
    key = cfg.posthog_api_key.get_secret_value()
    if not cfg.posthog_host.strip() or not cfg.posthog_project_id.strip() or not key:
        raise ExportError("training.posthog_host, posthog_project_id and posthog_api_key are unset")
    url = f"{cfg.posthog_host.rstrip('/')}/api/projects/{cfg.posthog_project_id.strip()}/query/"
    rows: list[dict[str, Any]] = []
    after: tuple[str, str] | None = None
    with httpx.Client(timeout=cfg.request_timeout_secs, transport=transport) as client:
        while True:
            page = _page(client, url, key, hogql(cfg.posthog_page_rows, after))
            rows.extend(page)
            if len(page) < cfg.posthog_page_rows:
                return rows
            last = page[-1]
            after = (str(last.get("timestamp")), str(last.get("uuid")))


def export_examples(
    cfg: Training | None = None, transport: httpx.BaseTransport | None = None
) -> list[TrainingExample]:
    """Every complete llm generation in PostHog as a TrainingExample."""
    examples = []
    for row in fetch_rows(cfg, transport):
        ex = row_to_example(row)
        if ex is not None:
            examples.append(ex)
    return examples
