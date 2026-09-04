//! Trace sinks: JSONL round trip.

use std::time::Duration;

use uuid::Uuid;

use crate::agent::harness::trace::JsonlSink;
use crate::core::traits::trace::TraceSink;
use crate::core::types::context::RequestContext;
use crate::core::types::model::Usage;
use crate::core::types::trace::{RunStatus, TraceEvent};

#[test]
fn jsonl_round_trips() {
    let dir = std::env::temp_dir().join(format!("sparky-trace-{}", Uuid::new_v4()));
    let Some(sink) = JsonlSink::new(&dir).ok() else {
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
    let records = sink.read(ctx.request_id).unwrap_or_default();
    assert_eq!(records.len(), 2);
    assert!(matches!(
        records[1].event,
        TraceEvent::Completed { steps: 1, .. }
    ));
    let _ = std::fs::remove_dir_all(dir);
}
