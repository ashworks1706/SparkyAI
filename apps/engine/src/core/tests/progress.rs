//! Live progress: which trace events reach the caller mid-run, and how they read on the wire.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::mpsc;

use crate::agent::harness::trace::Fanout;
use crate::core::tests::support::{MemorySink, ctx};
use crate::core::traits::trace::TraceSink;
use crate::core::types::context::RequestContext;
use crate::core::types::model::{FinishReason, Usage};
use crate::core::types::policy::Decision;
use crate::core::types::trace::{RunStatus, TraceEvent};
use crate::core::types::wire::Progress;

#[test]
fn events_the_caller_should_see_carry_their_own_wording() {
    let started = TraceEvent::ToolStarted {
        step: 1,
        tool: "public_search".into(),
    };
    assert_eq!(
        started.progress().as_deref(),
        Some("searching ASU pages"),
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
    assert_eq!(
        retrieval.progress().as_deref(),
        Some("found 2 passages for \"hayden hours\"")
    );
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

    // A client renders `text` and needs no match arm of its own; `event` is for the ones
    // that want to special-case a kind they already know.
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
            tool: "public_search".into(),
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
    assert_eq!(seen.as_deref(), Some("searching ASU pages"));
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
