//! Trace sinks: JSONL round trip.

use std::time::Duration;

use uuid::Uuid;

use crate::agent::harness::trace::JsonlSink;
use crate::core::traits::trace::TraceSink;
use crate::core::types::context::RequestContext;
use crate::core::types::model::Usage;
use crate::core::types::trace::{RunStatus, TraceEvent, TraceRecord};

#[test]
fn jsonl_round_trips() {
    let dir = std::env::temp_dir().join(format!("sparky-trace-{}", Uuid::new_v4()));
    let created = JsonlSink::new(&dir);
    assert!(created.is_ok(), "{created:?}");
    let Ok(sink) = created else {
        return;
    };
    let ctx = RequestContext::new("g", "u", Duration::from_secs(1));
    sink.emit(
        &ctx,
        TraceEvent::RequestStarted {
            input: "hi".into(),
            tenant_id: "g".into(),
            user_id: "u".into(),
        },
    );
    sink.emit(
        &ctx,
        TraceEvent::Completed {
            status: RunStatus::Answered,
            steps: 1,
            usage: Usage::default(),
            cost_usd: 0.0,
            duration_ms: 3,
        },
    );
    let text = std::fs::read_to_string(sink.path_for(ctx.request_id)).unwrap_or_default();
    let records: Vec<TraceRecord> = text
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    assert_eq!(records.len(), 2);
    assert!(matches!(
        records[1].event,
        TraceEvent::Completed { steps: 1, .. }
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn traceparent_parses_into_a_remote_parent() {
    use opentelemetry::trace::TraceContextExt;

    use crate::routes::chat::parse_traceparent;

    let cx = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
    let sc = cx.as_ref().map(|c| c.span().span_context().clone());
    assert!(sc.as_ref().is_some_and(|s| s.is_remote() && s.is_sampled()));
    assert_eq!(
        sc.map(|s| s.trace_id().to_string()),
        Some("4bf92f3577b34da6a3ce929d0e0e4736".into())
    );
    assert!(parse_traceparent("garbage").is_none());
    assert!(parse_traceparent("00-00000000000000000000000000000000-0000000000000000-01").is_none());
}
