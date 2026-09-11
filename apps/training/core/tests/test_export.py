import json

import httpx
import pytest
from pydantic import SecretStr
from training.core.settings import Training
from training.core.types import ExportError
from training.datasets.export import export_examples, hogql, row_to_example

_COLUMNS = ["uuid", "timestamp", "attributes"]
_U1 = "0191a000-0000-7000-8000-000000000001"
_U2 = "0191a000-0000-7000-8000-000000000002"
_U3 = "0191a000-0000-7000-8000-000000000003"

_INPUT = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
_OUTPUT = [{"role": "assistant", "content": "hello"}]


def _attrs(inp=_INPUT, out=_OUTPUT, span="llm", **extra):
    attrs = {
        "sparky.span": span,
        "gen_ai.input.messages": json.dumps(inp) if isinstance(inp, list) else inp,
        "gen_ai.output.messages": json.dumps(out) if isinstance(out, list) else out,
        "gen_ai.request.model": "m",
        "$ai_session_id": "s1",
        "posthog.distinct_id": "u1",
        "sparky.tools": json.dumps([{"name": "a"}, {"name": "b"}]),
    }
    attrs.update(extra)
    return attrs


def _row(attrs, row_id=_U1, ts="2026-09-11T10:00:00.123456Z", as_string=False):
    return {
        "uuid": row_id,
        "timestamp": ts,
        "attributes": json.dumps(attrs) if as_string else attrs,
    }


def _cfg(rows=2):
    return Training(
        posthog_host="http://ph:8010/",
        posthog_project_id="7",
        posthog_api_key=SecretStr("phx_key"),
        posthog_page_rows=rows,
    )


def test_span_attributes_become_an_example():
    ex = row_to_example(_row(_attrs()))

    assert ex is not None
    assert ex.id == _U1 and ex.model == "m" and ex.session_id == "s1" and ex.user_id == "u1"
    assert ex.tool_count == 2
    assert [m.role for m in ex.messages] == ["system", "user"]
    assert ex.response.content == "hello"


def test_attributes_as_a_json_string_are_accepted():
    ex = row_to_example(_row(_attrs(), as_string=True))

    assert ex is not None and ex.response.content == "hello"


def test_parsed_message_lists_are_accepted():
    ex = row_to_example(_row(_attrs(inp=_INPUT, out=_OUTPUT) | {"gen_ai.input.messages": _INPUT}))

    assert ex is not None and [m.role for m in ex.messages] == ["system", "user"]


def test_parts_messages_are_converted():
    inp = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "a"}, {"type": "text", "content": "b"}],
        }
    ]
    out = [{"role": "assistant", "parts": [{"type": "text", "content": "ok"}]}]

    ex = row_to_example(_row(_attrs(inp=inp, out=out)))

    assert ex is not None
    assert ex.messages[0].content == "ab" and ex.response.content == "ok"


def test_openai_text_content_lists_are_converted():
    inp = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    ex = row_to_example(_row(_attrs(inp=inp)))

    assert ex is not None and ex.messages[0].content == "hi"


def test_tool_calls_survive():
    out = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "t", "arguments": "{}"}],
        }
    ]

    ex = row_to_example(_row(_attrs(out=out)))

    assert ex is not None and ex.response.tool_calls[0]["name"] == "t"


def test_non_llm_and_replyless_rows_are_skipped():
    assert row_to_example(_row(_attrs(span="task"))) is None
    assert row_to_example(_row(_attrs(out=None))) is None
    assert row_to_example(_row(_attrs(out=[]))) is None


@pytest.mark.parametrize(
    "attrs",
    [
        _attrs(inp="not json"),
        _attrs(inp={"role": "user"}),
        _attrs(inp=["text"]),
        _attrs(inp=[{"role": "robot", "content": "x"}]),
        _attrs(out=[{"role": "assistant", "parts": [{"type": "image", "url": "x"}]}]),
        _attrs(inp=[{"role": "user", "content": [{"type": "image_url"}]}]),
        _attrs(**{"sparky.tools": "not json"}),
        _attrs(**{"sparky.tools": {"n": 1}}),
    ],
)
def test_malformed_llm_rows_raise(attrs):
    with pytest.raises(ExportError):
        row_to_example(_row(attrs))


def test_query_reads_llm_spans_in_keyset_order():
    first = hogql(10)
    assert "FROM posthog.trace_spans" in first
    assert "name = 'llm'" in first
    assert first.endswith("ORDER BY timestamp ASC, uuid ASC LIMIT 10")
    assert "OFFSET" not in first

    later = hogql(10, ("2026-09-11T10:00:00.123456Z", _U2))
    assert "timestamp > toDateTime('2026-09-11 10:00:00.123456', 'UTC')" in later
    assert f"uuid > '{_U2}'" in later


def test_cursor_values_are_validated():
    with pytest.raises(ExportError):
        hogql(10, ("2026-09-11", "x' OR 1=1 --"))


def test_pages_follow_the_keyset_cursor():
    pages = [
        [_row(_attrs(), _U1, "2026-09-11T10:00:00Z"), _row(_attrs(), _U2, "2026-09-11T10:00:01Z")],
        [_row(_attrs(), _U3, "2026-09-11T10:00:02Z")],
    ]
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://ph:8010/api/projects/7/query/"
        assert request.headers["authorization"] == "Bearer phx_key"
        body = json.loads(request.content)
        assert body["query"]["kind"] == "HogQLQuery"
        seen.append(body["query"]["query"])
        rows = pages[len(seen) - 1]
        return httpx.Response(
            200, json={"columns": _COLUMNS, "results": [[r[c] for c in _COLUMNS] for r in rows]}
        )

    examples = export_examples(_cfg(rows=2), transport=httpx.MockTransport(handler))

    assert [e.id for e in examples] == [_U1, _U2, _U3]
    assert len(seen) == 2
    assert "uuid >" not in seen[0]
    assert "toDateTime('2026-09-11 10:00:01.000000', 'UTC')" in seen[1] and _U2 in seen[1]


def test_error_status_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(403, text="forbidden"))

    with pytest.raises(ExportError, match="403"):
        export_examples(_cfg(), transport=transport)


def test_unexpected_response_shape_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"rows": []}))

    with pytest.raises(ExportError):
        export_examples(_cfg(), transport=transport)


def test_missing_settings_raise():
    with pytest.raises(ExportError):
        export_examples(Training(posthog_project_id=""))
