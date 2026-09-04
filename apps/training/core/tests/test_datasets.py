import json

import pytest
from training.core.types import ExportError, Message, TrainingExample
from training.datasets.export import span_to_example
from training.datasets.redact import redact_text
from training.datasets.verify import verify


def _span(name="llm", inp=None, out=None):
    attrs = {
        "input": {"value": json.dumps(inp) if inp is not None else None},
        "output": {"value": json.dumps(out) if out is not None else None},
        "llm": {"model_name": "m", "invocation_parameters": json.dumps({"tools": 2})},
        "session": {"id": "s1"},
        "user": {"id": "u1"},
    }
    return {
        "name": name,
        "attributes": json.dumps(attrs),
        "context": {"spanId": "abc", "traceId": "t"},
    }


def test_llm_span_becomes_example():
    ex = span_to_example(
        _span(
            inp=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
            out={"role": "assistant", "content": "hello"},
        )
    )
    assert ex is not None
    assert ex.model == "m" and ex.tool_count == 2 and ex.session_id == "s1"
    assert [m.role for m in ex.messages] == ["system", "user"]


def test_non_llm_or_incomplete_spans_are_skipped():
    assert span_to_example(_span(name="tool")) is None
    assert span_to_example(_span(inp=[{"role": "user", "content": "x"}], out=None)) is None


def test_malformed_llm_span_is_an_error_not_a_skip():
    span = _span(inp=[{"role": "user", "content": "x"}], out={"role": "assistant", "content": "y"})
    attrs = json.loads(span["attributes"])
    attrs["llm"]["invocation_parameters"] = "not json"
    span["attributes"] = json.dumps(attrs)
    with pytest.raises(ExportError):
        span_to_example(span)


def test_redaction_covers_email_phone_ids_tokens():
    text = (
        "mail a@b.edu call 480-555-1234 id 897822597377581086 "
        "tok AAAAAAAAAAAAAAAAAAAAAAAA.BBBBBB.CCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    )
    out = redact_text(text)
    assert "a@b.edu" not in out and "[email]" in out
    assert "555-1234" not in out and "[phone]" in out
    assert "897822597377581086" not in out and "[discord-id]" in out
    assert "BBBBBB" not in out and "[token]" in out


def _ex(id_, content="hello", first="system"):
    return TrainingExample(
        id=id_,
        messages=[Message(role=first, content="s"), Message(role="user", content="q")],
        response=Message(role="assistant", content=content),
    )


def test_verify_drops_bad_and_duplicate_examples():
    kept, reasons = verify([_ex("a"), _ex("b"), _ex("c", content=""), _ex("d", first="user")])
    assert [e.id for e in kept] == ["a"]
    assert reasons == {"duplicate": 1, "empty response": 1, "first message is not system": 1}
