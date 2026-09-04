//! `ModelRequest`, `ModelResponse`, `Usage`, `FinishReason`, `ModelError`.

use serde::{Deserialize, Serialize};

use crate::core::types::message::{Message, ToolCall};
use crate::core::types::tool::ToolDefinition;

/// One completion request. Provider JSON stays inside the adapter.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// Full assembled context, oldest first.
    pub messages: Vec<Message>,
    /// Tools the model may call this step.
    pub tools: Vec<ToolDefinition>,
    /// Completion budget.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
}

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of the answer.
    Stop,
    /// Wants tools executed.
    ToolCalls,
    /// Hit `max_tokens`.
    Length,
    /// Provider reported something else.
    Other,
}

/// Token counts for one call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub completion_tokens: u32,
}

impl Usage {
    /// Sum of both sides.
    pub fn total(self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }

    /// Adds another call's usage into this one.
    pub fn add(&mut self, other: Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
    }
}

/// One completion.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// Text the model produced. Empty when it only called tools.
    pub content: String,
    /// Tool calls, in the order the model listed them.
    pub tool_calls: Vec<ToolCall>,
    /// Why it stopped.
    pub finish_reason: FinishReason,
    /// Tokens consumed.
    pub usage: Usage,
    /// Model name as reported by the provider.
    pub model: String,
}

impl ModelResponse {
    /// The assistant turn to append to history.
    pub fn as_message(&self) -> Message {
        if self.tool_calls.is_empty() {
            Message::assistant(self.content.clone())
        } else {
            Message::assistant_tool_calls(self.content.clone(), self.tool_calls.clone())
        }
    }
}

/// Model call failures.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// Network or server failure; safe to retry.
    #[error("model transport: {0}")]
    Transport(String),
    /// The provider returned an error status.
    #[error("model returned {status}: {body}")]
    Status {
        /// HTTP status.
        status: u16,
        /// Response body, truncated.
        body: String,
    },
    /// Response could not be parsed into `ModelResponse`.
    #[error("model response malformed: {0}")]
    Malformed(String),
    /// The request deadline passed.
    #[error("model call timed out")]
    Timeout,
    /// The request was cancelled.
    #[error("model call cancelled")]
    Cancelled,
}

impl ModelError {
    /// Whether the loop should retry this failure.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Status { status, .. } => *status >= 500 || *status == 429,
            Self::Malformed(_) | Self::Timeout | Self::Cancelled => false,
        }
    }
}
