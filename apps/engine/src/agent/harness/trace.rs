//! `TraceSink` trait, `TraceEvent`, JSONL sink, replay.

use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::agent::harness::model::{FinishReason, Usage};
use crate::agent::harness::policy::Decision;
use crate::core::types::context::RequestContext;

/// One thing that happened during a request. Never carries secrets or raw credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceEvent {
    /// The loop began.
    RequestStarted {
        /// User input, as received.
        input: String,
        /// Tenant scope.
        tenant_id: String,
        /// Caller.
        user_id: String,
    },
    /// Context assembled for one step.
    ContextAssembled {
        /// Loop step, from 1.
        step: u32,
        /// Messages sent.
        message_count: usize,
        /// Rough prompt token estimate.
        estimated_tokens: usize,
        /// Evidence chunk ids included.
        evidence_ids: Vec<Uuid>,
    },
    /// One completion returned.
    ModelCall {
        /// Loop step.
        step: u32,
        /// Model name as reported.
        model: String,
        /// Why it stopped.
        finish_reason: FinishReason,
        /// Tokens.
        usage: Usage,
        /// Wall time.
        duration_ms: u64,
        /// Retry index, 0 for the first attempt.
        attempt: u32,
    },
    /// A model call failed.
    ModelError {
        /// Loop step.
        step: u32,
        /// Attempt that failed.
        attempt: u32,
        /// Error text.
        error: String,
        /// Whether the loop retried.
        retried: bool,
    },
    /// Policy ruled on a proposed tool call.
    PolicyDecision {
        /// Loop step.
        step: u32,
        /// Tool name.
        tool: String,
        /// Verdict.
        decision: Decision,
    },
    /// A tool ran.
    ToolCall {
        /// Loop step.
        step: u32,
        /// Provider call id.
        call_id: String,
        /// Tool name.
        tool: String,
        /// Validated arguments.
        arguments: Value,
        /// `Ok` text or `Err` message, truncated.
        result: Result<String, String>,
        /// Wall time.
        duration_ms: u64,
    },
    /// Retrieval ran.
    Retrieval {
        /// Loop step.
        step: u32,
        /// Query text.
        query: String,
        /// Chunk ids returned, best first.
        chunk_ids: Vec<Uuid>,
        /// Wall time.
        duration_ms: u64,
    },
    /// The loop finished.
    Completed {
        /// How it ended.
        status: RunStatus,
        /// Total steps.
        steps: u32,
        /// Total tokens.
        usage: Usage,
        /// Estimated cost in USD.
        cost_usd: f64,
        /// Wall time.
        duration_ms: u64,
    },
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// A final answer was produced.
    Answered,
    /// Stopped to ask the user to confirm an action.
    AwaitingConfirmation,
    /// Hit the step limit.
    StepLimit,
    /// Hit the deadline.
    Deadline,
    /// Cancelled by the caller.
    Cancelled,
    /// Failed with an error.
    Error,
}

/// A trace event with its envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Request this belongs to.
    pub request_id: Uuid,
    /// Conversation this belongs to.
    pub conversation_id: Uuid,
    /// When it was emitted.
    pub at: DateTime<Utc>,
    /// The event.
    #[serde(flatten)]
    pub event: TraceEvent,
}

/// Where trace events go. Must not block the loop.
pub trait TraceSink: Send + Sync {
    /// Records one event.
    fn emit(&self, ctx: &RequestContext, event: TraceEvent);
}

impl TraceSink for () {
    fn emit(&self, _ctx: &RequestContext, _event: TraceEvent) {}
}

/// Appends one JSON line per event to `<dir>/<request_id>.jsonl`.
#[derive(Debug)]
pub struct JsonlSink {
    dir: PathBuf,
}

impl JsonlSink {
    /// Writes under `dir`, creating it if missing.
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Path of the trace file for a request.
    pub fn path_for(&self, request_id: Uuid) -> PathBuf {
        self.dir.join(format!("{request_id}.jsonl"))
    }

    /// Reads every record of one request, in order.
    pub fn read(&self, request_id: Uuid) -> std::io::Result<Vec<TraceRecord>> {
        let text = std::fs::read_to_string(self.path_for(request_id))?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(std::io::Error::other))
            .collect()
    }
}

impl TraceSink for JsonlSink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        use std::io::Write;
        let record = TraceRecord {
            request_id: ctx.request_id,
            conversation_id: ctx.conversation_id,
            at: Utc::now(),
            event,
        };
        let Ok(line) = serde_json::to_string(&record) else {
            return;
        };
        let path = self.path_for(ctx.request_id);
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| writeln!(f, "{line}"));
        if let Err(e) = result {
            tracing::warn!(error = %e, path = %path.display(), "trace write failed");
        }
    }
}

/// Keeps every record in memory. For tests and for the admin trace viewer.
#[derive(Debug, Default)]
pub struct MemorySink {
    records: Mutex<Vec<TraceRecord>>,
}

impl MemorySink {
    /// Everything emitted so far.
    pub fn records(&self) -> Vec<TraceRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

impl TraceSink for MemorySink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        if let Ok(mut records) = self.records.lock() {
            records.push(TraceRecord {
                request_id: ctx.request_id,
                conversation_id: ctx.conversation_id,
                at: Utc::now(),
                event,
            });
        }
    }
}

/// Fans one event out to several sinks.
pub struct MultiSink(pub Vec<std::sync::Arc<dyn TraceSink>>);

impl TraceSink for MultiSink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        for sink in &self.0 {
            sink.emit(ctx, event.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn jsonl_round_trips() {
        let dir = std::env::temp_dir().join(format!("sparky-trace-{}", Uuid::new_v4()));
        let sink = JsonlSink::new(&dir).ok();
        let Some(sink) = sink else {
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
}
