//! Live progress: which trace events reach the caller mid-run, and how they read on the wire.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::mpsc;

use crate::agent::harness::trace::Fanout;
use crate::core::tests::support::{MemorySink, ctx};
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{FinishReason, Usage};
use crate::core::types::safety::policy::Decision;
use crate::core::types::trace::progress::Progress;
use crate::core::types::trace::{RunStatus, TraceEvent};

#[test]
fn events_the_caller_should_see_carry_their_own_wording() {
    let started = TraceEvent::ToolStarted {
        step: 1,
        tool: "search_knowledge_base".into(),
    };
    assert_eq!(
        started.progress().as_deref(),
        Some("searching the knowledge base"),
        "a tool the engine ships gets wording a student understands"
    );

    let unknown = TraceEvent::ToolStarted {
        step: 1,
        tool: "some_new_mcp_tool".into(),
    };
    assert_eq!(
        unknown.progress().as_deref(),
        Some("running some_new_mcp_tool"),
        "a tool nobody wrote a phrase for still reports something"
    );

    let denied = TraceEvent::PolicyDecision {
        step: 1,
        tool: "browser_click".into(),
        decision: Decision::Deny {
            reason: "no".into(),
        },
    };
    assert_eq!(
        denied.progress().as_deref(),
        Some("browser_click was not allowed")
    );

    let retrieval = TraceEvent::Retrieval {
        step: 1,
        query: "hayden hours".into(),
        chunk_ids: vec![uuid::Uuid::new_v4(), uuid::Uuid::new_v4()],
        duration_ms: 12,
    };
    assert_eq!(retrieval.progress().as_deref(), Some("reading 2 sources"));
}

#[test]
fn a_model_call_a_finished_tool_and_recalled_memory_each_get_a_line() {
    assert_eq!(
        TraceEvent::ModelStarted { step: 1 }.progress().as_deref(),
        Some("thinking")
    );

    let tool = |name: &str, result: Result<String, String>| TraceEvent::ToolCall {
        step: 1,
        call_id: "c1".into(),
        tool: name.into(),
        arguments: json!({}),
        result,
        duration_ms: 3,
    };
    assert_eq!(
        tool("search_knowledge_base", Ok("three passages".into()))
            .progress()
            .as_deref(),
        Some("knowledge base search finished"),
        "the line names the tool, never the result"
    );
    assert_eq!(
        tool("query_source", Err("upstream said 500".into()))
            .progress()
            .as_deref(),
        Some("live ASU page check failed"),
        "the line never carries the error text"
    );
    assert_eq!(
        tool("some_new_mcp_tool", Ok(String::new()))
            .progress()
            .as_deref(),
        Some("some_new_mcp_tool finished")
    );

    assert_eq!(
        TraceEvent::MemoryRecalled { count: 3 }
            .progress()
            .as_deref(),
        Some("remembering 3 things about you")
    );
    assert_eq!(
        TraceEvent::MemoryRecalled { count: 1 }
            .progress()
            .as_deref(),
        Some("remembering 1 thing about you")
    );
    assert!(TraceEvent::MemoryRecalled { count: 0 }.progress().is_none());

    let one = TraceEvent::Retrieval {
        step: 1,
        query: "q".into(),
        chunk_ids: vec![uuid::Uuid::new_v4()],
        duration_ms: 1,
    };
    assert_eq!(one.progress().as_deref(), Some("reading 1 source"));
}

#[test]
fn bookkeeping_events_stay_out_of_the_callers_way() {
    let assembled = TraceEvent::ContextAssembled {
        step: 1,
        message_count: 4,
        estimated_tokens: 900,
        evidence_ids: Vec::new(),
    };
    assert!(assembled.progress().is_none());

    let completed = TraceEvent::Completed {
        status: RunStatus::Answered,
        steps: 2,
        usage: Usage::default(),
        cost_usd: 0.0,
        duration_ms: 10,
    };
    assert!(completed.progress().is_none(), "the answer says this");
}

#[test]
fn the_wire_form_reads_without_knowing_the_variant() {
    let event = TraceEvent::ToolStarted {
        step: 2,
        tool: "browser_navigate".into(),
    };
    let Some(progress) = Progress::of(&event) else {
        unreachable!("ToolStarted is user-visible")
    };
    let wire = serde_json::to_value(&progress).unwrap_or(json!(null));

    // A client renders text and needs no match arm of its own; event names the kind.
    assert_eq!(wire["text"], json!("opening the page"));
    assert_eq!(wire["event"], json!("tool_started"));
}

#[tokio::test]
async fn a_run_with_a_listener_records_and_reports_at_once() {
    let sink = Arc::new(MemorySink::new());
    let (tx, mut rx) = mpsc::unbounded_channel();
    let fanout = Fanout::new(sink.clone());
    let listening = ctx().listening_to(tx);

    fanout.emit(
        &listening,
        TraceEvent::ToolStarted {
            step: 1,
            tool: "search_knowledge_base".into(),
        },
    );
    fanout.emit(
        &listening,
        TraceEvent::ContextAssembled {
            step: 1,
            message_count: 1,
            estimated_tokens: 1,
            evidence_ids: Vec::new(),
        },
    );

    assert_eq!(sink.records().len(), 2, "every event is still traced");
    let seen = rx.try_recv().ok().map(|p: Progress| p.text);
    assert_eq!(seen.as_deref(), Some("searching the knowledge base"));
    assert!(rx.try_recv().is_err(), "only user-visible events are sent");
}

#[tokio::test]
async fn a_run_nobody_is_watching_still_traces() {
    let sink = Arc::new(MemorySink::new());
    let fanout = Fanout::new(sink.clone());
    let quiet = RequestContext::new("g", "u", Duration::from_secs(5));

    fanout.emit(
        &quiet,
        TraceEvent::ModelCall {
            step: 1,
            model: "test".into(),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            duration_ms: 1,
            attempt: 0,
        },
    );

    assert_eq!(sink.records().len(), 1);
}
