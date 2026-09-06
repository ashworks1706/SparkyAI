//! `AgentConfig`, `Answer`, `AgentError`.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::core::types::assemble::Budget;
use crate::core::types::evidence::Evidence;
use crate::core::types::model::{ModelError, Usage};
use crate::core::types::policy::ConfirmationRequest;
use crate::core::types::tool::ToolRun;
use crate::core::types::trace::RunStatus;

/// Knobs for the loop. All bounded; nothing runs forever.
///
/// `Default` is implemented in `core::config`, where every one of these values is declared, so
/// a change to a setting cannot leave a second copy behind.
#[derive(Debug, Clone, Copy)]
pub struct AgentConfig {
    /// Maximum model calls per request.
    pub max_steps: u32,
    /// Retries on a retryable model error, per step.
    pub max_model_retries: u32,
    /// Per-tool-call timeout.
    pub tool_timeout: Duration,
    /// How long a held action waits for its caller's approval.
    pub confirmation_ttl: Duration,
    /// Completion budget per model call.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Evidence chunks to retrieve per request.
    pub retrieval_top_k: usize,
    /// Prior turns to load.
    pub history_turns: usize,
    /// Memories recalled per request.
    pub memory_recall_limit: usize,
    /// First retry wait, doubled per attempt.
    pub retry_base_ms: u64,
    /// Longest retry wait.
    pub retry_cap_ms: u64,
    /// Longest value recorded on a span; the JSONL trace keeps the rest.
    pub max_span_value_chars: usize,
    /// USD per million prompt tokens, for cost tracking. Zero for local models.
    pub usd_per_m_prompt: f64,
    /// USD per million completion tokens.
    pub usd_per_m_completion: f64,
    /// Prompt budgets.
    pub budget: Budget,
}

/// How a run ended and what it produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    /// Final text. Empty when awaiting confirmation.
    pub text: String,
    /// Evidence the answer was grounded in, best first.
    pub evidence: Vec<Evidence>,
    /// Set when the loop stopped to ask the user.
    pub confirmation: Option<ConfirmationRequest>,
    /// How it ended.
    pub status: RunStatus,
    /// Model calls made.
    pub steps: u32,
    /// Tools that ran, in order.
    pub tool_runs: Vec<ToolRun>,
    /// Tokens across every call.
    pub usage: Usage,
    /// Estimated cost in USD.
    pub cost_usd: f64,
}

/// Loop failures. Everything recoverable has already been fed back to the model.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The model failed after retries.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A store the request needs was unavailable.
    #[error("store: {0}")]
    Store(String),
}
