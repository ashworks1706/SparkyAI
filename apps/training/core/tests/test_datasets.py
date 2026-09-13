from training.core.types import Message, TrainingExample
from training.datasets.redact import redact_text
from training.datasets.verify import verify


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


def test_redaction_reaches_tool_call_arguments():
    """Redaction rewrites the string values inside tool call arguments."""
    from training.core.types import Message
    from training.datasets.redact import redact_message

    m = Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "c1",
                "name": "search_courses",
                "arguments": '{"query": "email me at student@asu.edu"}',
            }
        ],
    )

    out = redact_message(m)

    assert "student@asu.edu" not in str(out.tool_calls)
    assert "[email]" in str(out.tool_calls)
    assert out.tool_calls[0]["name"] == "search_courses", "only values are rewritten"
