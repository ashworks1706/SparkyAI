//! TraceEvent, RunStatus, TraceRecord.

pub mod progress;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::core::types::knowledge::cache::CacheOutcome;
use crate::core::types::knowledge::route::Skipped;
use crate::core::types::model::{FinishReason, Usage};
use crate::core::types::safety::guardrail::Stage;
use crate::core::types::safety::policy::Decision;
use crate::core::types::trace::progress::ProgressStyle;

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
    /// The model has reasoned this far on a step. Sent while it streams; never recorded.
    ModelReasoning {
        /// Loop step.
        step: u32,
        /// The reasoning so far.
        text: String,
    },
    /// What the model reasoned on a step, or wrote on its way to a tool call.
    ModelThought {
        /// Loop step.
        step: u32,
        /// What it wrote, truncated.
        text: String,
    },
    /// The answer as written so far, released a block at a time while it streams; never recorded.
    AnswerDraft {
        /// Loop step.
        step: u32,
        /// The answer so far.
        text: String,
    },
    /// The draft so far withdraws: tools were called, the call failed, or the guardrail refused it.
    AnswerDraftCleared {
        /// Loop step.
        step: u32,
    },
    /// The model wrote the answer and the loop is done.
    ModelAnswered {
        /// Loop step.
        step: u32,
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
        /// Provider call id.
        call_id: String,
        /// Tool name.
        tool: String,
        /// Validated arguments.
        arguments: Value,
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
    /// The router skipped retrieval for this question.
    RetrievalSkipped {
        /// Why it was skipped.
        reason: Skipped,
    },
    /// What the query cache did for a live query.
    QueryCache {
        /// Registry key of the source asked for.
        source: String,
        /// What the cache did.
        outcome: CacheOutcome,
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
    /// The snake case name this event serializes under.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::RequestStarted { .. } => "request_started",
            Self::ContextAssembled { .. } => "context_assembled",
            Self::ModelStarted { .. } => "model_started",
            Self::ModelCall { .. } => "model_call",
            Self::ModelReasoning { .. } => "model_reasoning",
            Self::ModelThought { .. } => "model_thought",
            Self::AnswerDraft { .. } => "answer_draft",
            Self::AnswerDraftCleared { .. } => "answer_draft_cleared",
            Self::ModelAnswered { .. } => "model_answered",
            Self::ModelError { .. } => "model_error",
            Self::PolicyDecision { .. } => "policy_decision",
            Self::GuardrailBlocked { .. } => "guardrail_blocked",
            Self::Compaction { .. } => "compaction",
            Self::ToolStarted { .. } => "tool_started",
            Self::ToolCall { .. } => "tool_call",
            Self::MemoryRecalled { .. } => "memory_recalled",
            Self::Retrieval { .. } => "retrieval",
            Self::RetrievalSkipped { .. } => "retrieval_skipped",
            Self::QueryCache { .. } => "query_cache",
            Self::Completed { .. } => "completed",
        }
    }

    /// What to show someone waiting on this run, or None when the event is bookkeeping.
    pub fn progress(&self, style: ProgressStyle) -> Option<String> {
        let detail = style.detail_chars;
        match self {
            Self::ModelStarted { .. } => Some("\u{1f914} thinking".to_owned()),
            Self::ModelReasoning { text, .. } => {
                let so_far = tail(&one_line(text), style.thought_chars);
                (!so_far.is_empty()).then(|| format!("\u{1f914} {so_far}"))
            }
            Self::ModelThought { text, .. } => {
                let thought = clip(&one_line(text), style.thought_chars);
                (!thought.is_empty()).then(|| format!("\u{1f4ad} {thought}"))
            }
            Self::AnswerDraft { text, .. } => {
                let draft = text.trim();
                (!draft.is_empty()).then(|| draft.to_owned())
            }
            Self::ToolStarted {
                tool, arguments, ..
            } => Some(format!(
                "\u{1f527} `{}`{} \u{2014} {}",
                tool,
                call_arguments(arguments, detail),
                running(tool)
            )),
            Self::ToolCall {
                tool,
                arguments,
                result,
                ..
            } => {
                let head = format!("`{}`{}", tool, call_arguments(arguments, detail));
                Some(match result {
                    Ok(output) => {
                        let shown = clip(&one_line(output), detail);
                        if shown.is_empty() {
                            format!("\u{2705} {head} \u{2014} nothing came back")
                        } else {
                            format!("\u{2705} {head} \u{2192} {shown}")
                        }
                    }
                    Err(error) => format!(
                        "\u{274c} {head} \u{2014} {}",
                        clip(&one_line(error), detail)
                    ),
                })
            }
            Self::MemoryRecalled { count: 0 } => None,
            Self::MemoryRecalled { count } => Some(format!(
                "\u{1f9e0} remembering {count} {} about you",
                plural(*count, "thing", "things")
            )),
            Self::GuardrailBlocked { stage, .. } => {
                Some(format!("\u{1f6d1} the {} was not allowed", stage.as_str()))
            }
            Self::Compaction { turns } => Some(format!(
                "\u{1f5dc}\u{fe0f} summarising {turns} earlier turns"
            )),
            Self::Retrieval { chunk_ids, .. } => Some(format!(
                "\u{1f4da} read {} {} from the knowledge base",
                chunk_ids.len(),
                plural(chunk_ids.len(), "source", "sources")
            )),
            Self::PolicyDecision { tool, decision, .. } => match decision {
                Decision::Deny { .. } => Some(format!("\u{1f6ab} `{tool}` was not allowed")),
                Decision::Confirm(_) => Some(format!("\u{270b} `{tool}` needs your approval")),
                Decision::Allow => None,
            },
            Self::ModelError { retried: true, .. } => {
                Some("\u{1f504} the model stumbled, retrying".to_owned())
            }
            // The tool call the skip leads to writes its own line, and so does a cached one.
            Self::QueryCache { .. }
            | Self::RetrievalSkipped { .. }
            | Self::RequestStarted { .. }
            | Self::ContextAssembled { .. }
            | Self::ModelCall { .. }
            | Self::ModelAnswered { .. }
            | Self::AnswerDraftCleared { .. }
            | Self::ModelError { .. }
            | Self::Completed { .. } => None,
        }
    }

    /// Whether this event removes the line of its slot instead of writing one.
    pub fn clears_slot(&self) -> bool {
        matches!(
            self,
            Self::ModelAnswered { .. } | Self::AnswerDraftCleared { .. }
        )
    }

    /// Whether this event carries the answer being written rather than a step.
    pub fn is_draft(&self) -> bool {
        matches!(
            self,
            Self::AnswerDraft { .. } | Self::AnswerDraftCleared { .. }
        )
    }

    /// Whether this event is sent only to watchers and left out of the recorded trace.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::ModelReasoning { .. }
                | Self::AnswerDraft { .. }
                | Self::AnswerDraftCleared { .. }
        )
    }

    /// Line an event overwrites, or None for a new one. Calls share start slot; events share step.
    pub fn slot(&self) -> Option<String> {
        match self {
            Self::ToolStarted { call_id, .. } | Self::ToolCall { call_id, .. } => {
                Some(format!("tool:{call_id}"))
            }
            Self::ModelStarted { step }
            | Self::ModelReasoning { step, .. }
            | Self::ModelThought { step, .. }
            | Self::ModelAnswered { step } => Some(format!("model:{step}")),
            Self::AnswerDraft { .. } | Self::AnswerDraftCleared { .. } => {
                Some(ANSWER_SLOT.to_owned())
            }
            _ => None,
        }
    }
}

/// A compact rendering of the arguments a call was made with. Empty when there are none.
fn call_arguments(arguments: &Value, limit: usize) -> String {
    let Some(fields) = arguments.as_object() else {
        return String::new();
    };
    let shown: Vec<String> = fields
        .iter()
        .map(|(key, value)| match value {
            Value::String(text) => format!("{key}: {text}"),
            other => format!("{key}: {other}"),
        })
        .collect();
    if shown.is_empty() {
        return String::new();
    }
    format!(" ({})", clip(&one_line(&shown.join(", ")), limit))
}

/// The slot the answer draft is written to.
pub const ANSWER_SLOT: &str = "answer";

/// The last limit characters of text, with an ellipsis in front when it was cut.
fn tail(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_owned();
    }
    let kept: String = text.chars().skip(count - limit.saturating_sub(1)).collect();
    format!("\u{2026}{}", kept.trim_start())
}

/// Text as one line, with runs of whitespace collapsed.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Text held to limit characters, with an ellipsis when it was cut.
fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let kept: String = text.chars().take(limit.saturating_sub(1)).collect();
    format!("{}\u{2026}", kept.trim_end())
}

/// What a tool is shown as while it runs. A search tool names what it searches.
fn running(tool: &str) -> String {
    if tool == "get_skill" {
        return "reading a saved procedure".to_owned();
    }
    match tool.strip_prefix("search_") {
        Some(source) => format!("searching live {}", source.replace('_', " ")),
        None => "running".to_owned(),
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
    /// What to tell the caller when the loop stopped with no answer. None for Answered and Blocked.
    pub fn explain(&self) -> Option<&'static str> {
        match self {
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
