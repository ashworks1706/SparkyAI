from training.core.types import EvalCase, Expectation, TurnResult
from training.evals.suites import (
    clarification,
    grounding,
    latency,
    memory,
    permissions,
    refusal,
    tool_args,
    tool_selection,
)


def _turn(text="", status="answered", citations=(), events=(), latency_ms=100, conv="c1"):
    return TurnResult(
        request_id="r",
        conversation_id=conv,
        status=status,
        text=text,
        citations=list(citations),
        steps=1,
        tokens=1,
        latency_ms=latency_ms,
        events=list(events),
    )


def _case(**expect):
    return EvalCase(id="x", suites=[], question="q", expect=Expectation(**expect))


def test_tool_selection_and_args():
    ev = [{"kind": "tool_call", "tool": "search_asu", "arguments": {"query": "Library hours"}}]
    assert tool_selection.score(_case(tool="search_asu"), [_turn(events=ev)]).passed
    assert not tool_selection.score(_case(tool="search_asu"), [_turn()]).passed
    assert not tool_selection.score(_case(), [_turn(events=ev)]).passed
    named = tool_args.score(
        _case(tool="search_asu", tool_args_contain={"query": "library"}), [_turn(events=ev)]
    )
    assert named.passed, "the key names the argument, the value is matched inside it"
    wrong_key = tool_args.score(
        _case(tool="search_asu", tool_args_contain={"q": "library"}), [_turn(events=ev)]
    )
    assert not wrong_key.passed, "an expectation naming an argument the tool lacks fails"
    assert tool_args.score(_case(tool="search_asu"), [_turn(events=ev)]) is None, (
        "no expectation means no result, not a free pass"
    )


def test_grounding_needs_citation_status_and_mentions():
    good = _turn(text="Open 7am to 9pm", citations=["library_hours — url"])
    assert grounding.score(_case(source_key="library_hours", mentions=["7am"]), [good]).passed
    assert not grounding.score(_case(source_key="events"), [good]).passed
    assert not grounding.score(_case(source_key="library_hours"), [_turn(status="stalled")]).passed


def test_refusal_and_clarification():
    assert refusal.score(
        _case(refuse=True), [_turn(text="I can't find that in my sources.")]
    ).passed
    assert not refusal.score(
        _case(refuse=True), [_turn(text="Sure, it is 5.", citations=["x"])]
    ).passed
    assert clarification.score(
        _case(clarify=True), [_turn(text="Which library do you mean?")]
    ).passed
    ev = [{"kind": "tool_call", "tool": "search_asu", "arguments": {}}]
    assert not clarification.score(_case(clarify=True), [_turn(text="Which?", events=ev)]).passed


def test_permissions_memory_latency():
    deny = [{"kind": "policy_decision", "tool": "t", "decision": {"decision": "deny"}}]
    assert permissions.score(_case(policy="deny"), [_turn(events=deny)]).passed
    assert permissions.score(
        _case(policy="confirm"), [_turn(status="awaiting_confirmation")]
    ).passed
    assert memory.score(
        _case(remembers="cs"), [_turn(text="hi"), _turn(text="You study CS.")]
    ).passed
    assert not memory.score(_case(remembers="cs"), [_turn(text="hi")]).passed
    assert latency.score(_case(max_latency_ms=200), [_turn(latency_ms=150)]).passed
    assert not latency.score(_case(max_latency_ms=100), [_turn(latency_ms=150)]).passed
