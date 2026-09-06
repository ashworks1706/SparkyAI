//! Wire types mirrored from the engine /chat contract, plus the client error.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// What the bot sends. Mirrors engine::core::types::chat::ChatRequest.
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

/// An action the engine is holding until the caller who asked answers it.
#[derive(Debug, Clone, Deserialize)]
pub struct Confirmation {
    /// Single-use token the buttons echo back so the engine can find the held action.
    pub token: Uuid,
    /// Tool that would have run.
    pub tool: String,
    /// What would happen.
    pub summary: String,
}

/// What the engine returns. Mirrors engine::core::types::chat::ChatResponse.
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

/// The engine error frame on /chat/stream.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorFrame {
    /// What went wrong.
    pub error: String,
    /// The status the JSON route would have returned.
    #[serde(default)]
    pub status: Option<u16>,
}

/// One line of progress from /chat/stream.
///
/// Only text is read. The engine renders the sentence.
#[derive(Debug, Clone, Deserialize)]
pub struct Progress {
    /// Ready-to-display sentence.
    pub text: String,
}

/// What arrives while a streamed turn runs.
#[derive(Debug)]
pub enum Update {
    /// Something happened worth showing.
    Progress(String),
    /// The turn finished.
    Answer(Box<ChatResponse>),
    /// The turn failed.
    Failed(EngineError),
}

/// Body of POST /confirm, answering an action the engine is holding.
#[derive(Debug, Serialize)]
pub struct ConfirmRequest {
    /// The token from the confirmation.
    pub token: Uuid,
    /// Whether to run it.
    pub approve: bool,
    /// Who is answering. The engine only accepts the caller who was asked.
    pub user_id: String,
    /// Guild scope.
    pub tenant_id: String,
    /// The conversation the held action belongs to.
    pub conversation_id: Uuid,
}
