import json

import httpx
import pytest
from pydantic import SecretStr
from training.core.settings import Telemetry, Training
from training.core.types import ExportError
from training.datasets.export import export_examples, span_to_example, spans_url

_S1 = "0191a0000000000001"
_S2 = "0191a0000000000002"
_S3 = "0191a0000000000003"

_INPUT = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
_OUTPUT = [{"role": "assistant", "content": "hello"}]


def _attrs(inp=_INPUT, out=_OUTPUT, span="llm", **extra):
    attrs = {
        "sparky.span": span,
        "gen_ai.input.messages": json.dumps(inp) if isinstance(inp, list) else inp,
        "gen_ai.output.messages": json.dumps(out) if isinstance(out, list) else out,
        "gen_ai.request.model": "m",
        "session.id": "s1",
        "user.id": "u1",
        "sparky.tools": json.dumps([{"name": "a"}, {"name": "b"}]),
    }
    attrs.update(extra)
    return attrs


def _span(attrs, span_id=_S1, name="llm"):
    return {
        "id": span_id,
        "name": name,
        "context": {"trace_id": "t1", "span_id": span_id},
        "attributes": attrs,
    }


def _telemetry():
    return Telemetry(
        phoenix_url="http://phoenix:6006/",
        phoenix_api_key=SecretStr("px_key"),
        project_name="sparky",
    )


def _cfg(spans=2):
    return Training(phoenix_page_spans=spans)


def test_span_attributes_become_an_example():
    ex = span_to_example(_span(_attrs()))

    assert ex is not None
    assert ex.id == _S1 and ex.model == "m" and ex.session_id == "s1" and ex.user_id == "u1"
    assert ex.tool_count == 2
    assert [m.role for m in ex.messages] == ["system", "user"]
    assert ex.response.content == "hello"


def test_nested_attributes_are_read_the_same_as_flat_ones():
    nested = {
        "sparky": {"span": "llm", "tools": json.dumps([{"name": "a"}])},
        "gen_ai": {
            "input": {"messages": json.dumps(_INPUT)},
            "output": {"messages": json.dumps(_OUTPUT)},
            "request": {"model": "m"},
        },
        "session": {"id": "s1"},
        "user": {"id": "u1"},
    }

    ex = span_to_example(_span(nested))

    assert ex is not None
    assert ex.model == "m" and ex.session_id == "s1" and ex.user_id == "u1"
    assert ex.tool_count == 1 and ex.response.content == "hello"


def test_the_span_id_falls_back_to_the_trace_context():
    span = _span(_attrs())
    del span["id"]

    ex = span_to_example(span)

    assert ex is not None and ex.id == _S1


def test_parsed_message_lists_are_accepted():
    ex = span_to_example(_span(_attrs() | {"gen_ai.input.messages": _INPUT}))

    assert ex is not None and [m.role for m in ex.messages] == ["system", "user"]


def test_parts_messages_are_converted():
    inp = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "a"}, {"type": "text", "content": "b"}],
        }
    ]
    out = [{"role": "assistant", "parts": [{"type": "text", "content": "ok"}]}]

    ex = span_to_example(_span(_attrs(inp=inp, out=out)))

    assert ex is not None
    assert ex.messages[0].content == "ab" and ex.response.content == "ok"


def test_openai_text_content_lists_are_converted():
    inp = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    ex = span_to_example(_span(_attrs(inp=inp)))

    assert ex is not None and ex.messages[0].content == "hi"


def test_tool_calls_survive():
    out = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "t", "arguments": "{}"}],
        }
    ]

    ex = span_to_example(_span(_attrs(out=out)))

    assert ex is not None and ex.response.tool_calls[0]["name"] == "t"


def test_non_llm_and_replyless_spans_are_skipped():
    assert span_to_example(_span(_attrs(span="task"))) is None
    assert span_to_example(_span(_attrs(out=None))) is None
    assert span_to_example(_span(_attrs(out=[]))) is None
    assert span_to_example(_span({}, name="retrieve")) is None


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
def test_malformed_llm_spans_raise(attrs):
    with pytest.raises(ExportError):
        span_to_example(_span(attrs))


def test_spans_are_read_from_the_project_endpoint():
    assert spans_url(_telemetry()) == "http://phoenix:6006/v1/projects/sparky/spans"


def test_pages_follow_the_cursor():
    pages = [
        ([_span(_attrs(), _S1), _span(_attrs(), _S2)], "cur1"),
        ([_span(_attrs(), _S3)], None),
    ]
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/projects/sparky/spans"
        assert request.headers["authorization"] == "Bearer px_key"
        assert request.url.params["name"] == "llm"
        assert request.url.params["limit"] == "2"
        seen.append(request.url.params.get("cursor"))
        data, cursor = pages[len(seen) - 1]
        return httpx.Response(200, json={"data": data, "next_cursor": cursor})

    examples = export_examples(_cfg(spans=2), _telemetry(), transport=httpx.MockTransport(handler))

    assert [e.id for e in examples] == [_S1, _S2, _S3]
    assert seen == [None, "cur1"]


def test_error_status_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(403, text="forbidden"))

    with pytest.raises(ExportError, match="403"):
        export_examples(_cfg(), _telemetry(), transport=transport)


def test_unexpected_response_shape_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"spans": []}))

    with pytest.raises(ExportError):
        export_examples(_cfg(), _telemetry(), transport=transport)


def test_missing_settings_raise():
    with pytest.raises(ExportError):
        export_examples(_cfg(), Telemetry(phoenix_url=""))
    with pytest.raises(ExportError):
        export_examples(_cfg(), Telemetry(phoenix_url="http://phoenix:6006", project_name=" "))
