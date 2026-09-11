//! TraceEvent, RunStatus, TraceRecord.

pub mod progress;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::core::types::model::{FinishReason, Usage};
use crate::core::types::safety::guardrail::Stage;
use crate::core::types::safety::policy::Decision;

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
    /// A model call is about to start. Once per step, whatever the retries.
    ModelStarted {
        /// Loop step.
        step: u32,
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
    /// The guardrail refused a response.
    GuardrailBlocked {
        /// Loop step.
        step: u32,
        /// Which branch the response was on.
        stage: Stage,
        /// Why it was refused.
        reason: String,
    },
    /// History that no longer fits is being replaced by one turn.
    Compaction {
        /// Turns being replaced.
        turns: usize,
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
        /// Ok text or Err message, truncated.
        result: Result<String, String>,
        /// Wall time.
        duration_ms: u64,
    },
    /// Memory and the profile graph were read for the prompt.
    MemoryRecalled {
        /// Memories recalled.
        count: usize,
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
    /// The snake-case name this event serialises under, for clients that switch on the kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::RequestStarted { .. } => "request_started",
            Self::ContextAssembled { .. } => "context_assembled",
            Self::ModelStarted { .. } => "model_started",
            Self::ModelCall { .. } => "model_call",
            Self::ModelError { .. } => "model_error",
            Self::PolicyDecision { .. } => "policy_decision",
            Self::GuardrailBlocked { .. } => "guardrail_blocked",
            Self::Compaction { .. } => "compaction",
            Self::ToolStarted { .. } => "tool_started",
            Self::ToolCall { .. } => "tool_call",
            Self::MemoryRecalled { .. } => "memory_recalled",
            Self::Retrieval { .. } => "retrieval",
            Self::Completed { .. } => "completed",
        }
    }

    /// What to show someone waiting on this run, or None when the event is bookkeeping.
    ///
    /// The match is exhaustive: a new event must decide whether it is shown.
    pub fn progress(&self) -> Option<String> {
        match self {
            Self::ModelStarted { .. } => Some("thinking".to_owned()),
            Self::ToolStarted { tool, .. } => Some(match friendly(tool) {
                Some((started, _)) => started.to_owned(),
                None => format!("running {tool}"),
            }),
            Self::ToolCall { tool, result, .. } => {
                let name = friendly(tool).map_or(tool.as_str(), |(_, name)| name);
                let outcome = if result.is_ok() { "finished" } else { "failed" };
                Some(format!("{name} {outcome}"))
            }
            Self::MemoryRecalled { count: 0 } => None,
            Self::MemoryRecalled { count } => Some(format!(
                "remembering {count} {} about you",
                plural(*count, "thing", "things")
            )),
            Self::GuardrailBlocked { stage, .. } => {
                Some(format!("the {} was not allowed", stage.as_str()))
            }
            Self::Compaction { turns } => Some(format!("summarising {turns} earlier turns")),
            Self::Retrieval { chunk_ids, .. } => Some(format!(
                "reading {} {}",
                chunk_ids.len(),
                plural(chunk_ids.len(), "source", "sources")
            )),
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
            | Self::Completed { .. } => None,
        }
    }
}

/// What a tool is shown as: the line while it runs, and its name once it is done. None for a
/// tool with no wording, which is shown by its own name.
fn friendly(tool: &str) -> Option<(&'static str, &'static str)> {
    match tool {
        "search_knowledge_base" => Some(("searching the knowledge base", "knowledge base search")),
        "browser_navigate" => Some(("opening the page", "page load")),
        "browser_snapshot" => Some(("reading the page", "page read")),
        "query_source" => Some(("checking a live ASU page", "live ASU page check")),
        _ => None,
    }
}

/// The singular form for a count of one, the plural otherwise.
fn plural(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 { one } else { many }
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
    /// The guardrail refused the response.
    Blocked,
    /// Failed with an error.
    Error,
}

impl RunStatus {
    /// What to tell the caller when the loop stopped without the model writing an answer.
    ///
    /// None only for Answered, where the model text is the answer. The match is exhaustive.
    pub fn explain(&self) -> Option<&'static str> {
        match self {
            // Answered carries the model text. Blocked carries the replacement the guardrail
            // supplied. Neither needs a stand-in here.
            Self::Answered | Self::Blocked => None,
            Self::AwaitingConfirmation => Some("I stopped to ask you first."),
            Self::StepLimit => Some("I could not finish within the allowed number of steps."),
            Self::Stalled => {
                Some("I kept repeating myself without getting further; try rephrasing.")
            }
            Self::Deadline => Some("That took too long, so I stopped."),
            Self::Cancelled => Some("Cancelled."),
            Self::Error => Some("Something went wrong before I could answer."),
        }
    }
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
