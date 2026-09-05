//! `TraceEvent`, `RunStatus`, `TraceRecord`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::core::types::model::{FinishReason, Usage};
use crate::core::types::policy::Decision;

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
    /// A tool is about to run.
    ToolStarted {
        /// Loop step.
        step: u32,
        /// Tool name.
        tool: String,
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

impl TraceEvent {
    /// What to show someone waiting on this run, or `None` when the event is bookkeeping.
    ///
    /// The match is exhaustive on purpose: a new event has to decide whether it is worth
    /// interrupting the caller for, rather than silently defaulting to hidden.
    pub fn progress(&self) -> Option<String> {
        match self {
            Self::ToolStarted { tool, .. } => Some(match tool.as_str() {
                "search_asu" | "public_search" => "searching ASU pages".to_owned(),
                "browser_navigate" => "opening the page".to_owned(),
                "browser_snapshot" => "reading the page".to_owned(),
                other => format!("running {other}"),
            }),
            Self::Retrieval {
                query, chunk_ids, ..
            } => Some(format!("found {} passages for {query:?}", chunk_ids.len())),
            Self::PolicyDecision { tool, decision, .. } => match decision {
                Decision::Deny { .. } => Some(format!("{tool} was not allowed")),
                Decision::Confirm(_) => Some(format!("{tool} needs your approval")),
                Decision::Allow => None,
            },
            Self::ModelError { retried: true, .. } => {
                Some("the model stumbled, retrying".to_owned())
            }
            Self::RequestStarted { .. }
            | Self::ContextAssembled { .. }
            | Self::ModelCall { .. }
            | Self::ModelError { .. }
            | Self::ToolCall { .. }
            | Self::Completed { .. } => None,
        }
    }
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
    /// Kept repeating the same tool calls without answering.
    Stalled,
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
