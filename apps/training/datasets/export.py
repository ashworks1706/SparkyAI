"""Pull llm spans from Phoenix and convert them to TrainingExamples."""

from __future__ import annotations

import json
from typing import Any

import httpx

from training.core.settings import Telemetry, Training, settings
from training.core.types import ExportError, Message, TrainingExample

_INPUT = "gen_ai.input.messages"
_OUTPUT = "gen_ai.output.messages"
_MODEL = "gen_ai.request.model"
_SESSION = "session.id"
_USER = "user.id"
_TOOLS = "sparky.tools"
_KIND = "sparky.span"

# Span name the engine gives one model call.
LLM_SPAN = "llm"

# The Phoenix spans endpoint under its base URL, by project.
SPANS_PATH = "/v1/projects/{project}/spans"


def spans_url(cfg: Telemetry) -> str:
    """Where spans of the configured project are read from."""
    host = cfg.phoenix_url.strip().rstrip("/")
    project = cfg.project_name.strip()
    if not host or not project:
        raise ExportError("telemetry.phoenix_url and telemetry.project_name must both be set")
    return host + SPANS_PATH.format(project=project)


def _headers(cfg: Telemetry) -> dict[str, str]:
    """Accept, plus a bearer token when Phoenix authenticates."""
    headers = {"accept": "application/json"}
    key = cfg.phoenix_api_key.get_secret_value().strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def attribute(attrs: Any, key: str) -> Any:
    """One attribute by its dotted name, whether Phoenix stored it flat or nested."""
    if not isinstance(attrs, dict):
        return None
    if key in attrs:
        return attrs[key]
    node: Any = attrs
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _json_value(value: Any, what: str, span_id: str) -> Any:
    """Parses a JSON string; any other value is returned as is."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ExportError(f"span {span_id}: {what} is not JSON: {e}") from e


def _text_of(items: list[Any], key: str, span_id: str) -> str:
    """Joins text items of shape {type: text, <key>: str}. Any other item raises."""
    texts = []
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("type") != "text"
            or not isinstance(item.get(key), str)
        ):
            raise ExportError(f"span {span_id}: unsupported message part {item!r:.200}")
        texts.append(item[key])
    return "".join(texts)


_SUMMARY_LEAD = "Summary of earlier turns: "


def _message(raw: Any, span_id: str) -> Message:
    """One message in the engine Message shape, a parts shape, or a text content list."""
    if not isinstance(raw, dict):
        raise ExportError(f"span {span_id}: message is not an object: {raw!r:.200}")
    data = dict(raw)
    if "parts" in data:
        parts = data.pop("parts")
        if not isinstance(parts, list) or "content" in data:
            raise ExportError(f"span {span_id}: bad parts message {raw!r:.200}")
        data["content"] = _text_of(parts, "content", span_id)
    elif isinstance(data.get("content"), list):
        data["content"] = _text_of(data["content"], "text", span_id)
    elif data.get("content") is None:
        data.pop("content", None)
    if data.get("role") == "summary":
        # The engine sends a summary to the model as a system message with this lead.
        data["role"] = "system"
        data["content"] = f"{_SUMMARY_LEAD}{data.get('content', '')}"
    try:
        return Message.model_validate(data)
    except ValueError as e:
        raise ExportError(f"span {span_id}: bad message: {e}") from e


def _messages(value: Any, what: str, span_id: str) -> list[Message]:
    parsed = _json_value(value, what, span_id)
    if not isinstance(parsed, list):
        raise ExportError(f"span {span_id}: {what} is not a list")
    return [_message(m, span_id) for m in parsed]


def _tool_count(value: Any, span_id: str) -> int:
    """sparky.tools as a count: a list of tool definitions, its JSON, or an integer."""
    if value is None:
        return 0
    parsed = _json_value(value, _TOOLS, span_id)
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return parsed
    raise ExportError(f"span {span_id}: bad {_TOOLS} {value!r:.200}")


def _span_id(span: dict[str, Any]) -> str:
    """The id of a span, from its own field or its trace context."""
    for candidate in (span.get("id"), (span.get("context") or {}).get("span_id")):
        if isinstance(candidate, str) and candidate:
            return candidate
    raise ExportError(f"span without an id: {span!r:.200}")


def span_to_example(span: dict[str, Any]) -> TrainingExample | None:
    """Maps a Phoenix span to an example; skips non-llm and unfinished ones, raises on malformed."""
    span_id = _span_id(span)
    attrs = span.get("attributes")
    if attrs is None:
        return None
    if not isinstance(attrs, dict):
        raise ExportError(f"span {span_id}: attributes is not an object")
    kind = attribute(attrs, _KIND)
    if kind is not None:
        if kind != LLM_SPAN:
            return None
    elif span.get("name") != LLM_SPAN:
        return None
    inp = attribute(attrs, _INPUT)
    out = attribute(attrs, _OUTPUT)
    if inp in (None, "", []) or out in (None, "", []):
        return None
    messages = _messages(inp, _INPUT, span_id)
    choices = _messages(out, _OUTPUT, span_id)
    if not choices:
        return None
    session = attribute(attrs, _SESSION)
    user = attribute(attrs, _USER)
    return TrainingExample(
        id=span_id,
        messages=messages,
        response=choices[0],
        model=attribute(attrs, _MODEL),
        session_id=str(session) if session else None,
        user_id=str(user) if user else None,
        tool_count=_tool_count(attribute(attrs, _TOOLS), span_id),
    )


def _page(
    client: httpx.Client, url: str, headers: dict[str, str], params: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    """One page of spans and the cursor of the next, which is None at the end."""
    try:
        r = client.get(url, headers=headers, params=params)
    except httpx.HTTPError as e:
        raise ExportError(f"phoenix request failed: {e}") from e
    if r.status_code != 200:
        raise ExportError(f"phoenix returned {r.status_code}: {r.text[:200]}")
    try:
        body = r.json()
        data = body["data"]
    except (ValueError, KeyError, TypeError) as e:
        raise ExportError(f"unexpected spans response shape: {e}") from e
    if not isinstance(data, list):
        raise ExportError("unexpected spans response shape: data is not a list")
    cursor = body.get("next_cursor")
    return data, cursor if isinstance(cursor, str) and cursor else None


def fetch_spans(
    cfg: Training | None = None,
    telemetry: Telemetry | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[dict[str, Any]]:
    """Every llm span of the project, paged through the Phoenix spans endpoint."""
    loaded = settings()
    cfg = cfg or loaded.training
    telemetry = telemetry or loaded.telemetry
    url = spans_url(telemetry)
    headers = _headers(telemetry)
    spans: list[dict[str, Any]] = []
    cursor: str | None = None
    with httpx.Client(timeout=cfg.request_timeout_secs, transport=transport) as client:
        while True:
            params: dict[str, Any] = {"limit": cfg.phoenix_page_spans, "name": LLM_SPAN}
            if cursor:
                params["cursor"] = cursor
            page, cursor = _page(client, url, headers, params)
            spans.extend(page)
            if cursor is None or not page:
                return spans


def export_examples(
    cfg: Training | None = None,
    telemetry: Telemetry | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[TrainingExample]:
    """Every complete llm generation in Phoenix as a TrainingExample."""
    examples = []
    for span in fetch_spans(cfg, telemetry, transport):
        ex = span_to_example(span)
        if ex is not None:
            examples.append(ex)
    return examples
