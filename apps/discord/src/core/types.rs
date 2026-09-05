//! Wire types mirrored from the engine's `/chat` contract, plus the client error.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// What the bot sends. Mirrors `engine::core::types::chat::ChatRequest`.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
    /// Channel id.
    pub channel_id: String,
    /// Role names the member holds.
    pub roles: Vec<String>,
    /// Continue this conversation, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<Uuid>,
    /// The question.
    pub message: String,
}

/// A pending confirmation the engine wants the user to approve. The token is not carried
/// because the bot cannot relay approvals until the Phase 3 confirm endpoint exists.
#[derive(Debug, Clone, Deserialize)]
pub struct Confirmation {
    /// Tool that would have run.
    pub tool: String,
    /// What would happen.
    pub summary: String,
}

/// What the engine returns. Mirrors `engine::core::types::chat::ChatResponse`.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    /// Trace id.
    pub request_id: Uuid,
    /// Conversation to continue with.
    pub conversation_id: Uuid,
    /// The answer.
    pub text: String,
    /// Citation lines.
    #[serde(default)]
    pub citations: Vec<String>,
    /// Set when the engine stopped to ask.
    #[serde(default)]
    pub confirmation: Option<Confirmation>,
    /// How the run ended.
    pub status: String,
    /// Tools the agent ran, in order.
    #[serde(default)]
    pub tools: Vec<ToolRun>,
}

/// One tool the agent ran.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolRun {
    /// Tool name.
    pub tool: String,
    /// Whether it returned a result.
    pub ok: bool,
}

/// Engine call failures.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Could not reach the engine.
    #[error("engine unreachable: {0}")]
    Transport(String),
    /// The engine answered with an error status.
    #[error("engine returned {status}: {body}")]
    Status {
        /// HTTP status.
        status: u16,
        /// Body, truncated.
        body: String,
    },
}
