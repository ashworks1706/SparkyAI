//! The chat turn: request, response, sources, and the held action.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::conversation::{Attachment, FileAttachment, Visibility};

/// What the bot sends. Mirrors engine::core::types::chat::ChatRequest.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
    /// Channel or thread id the turn is anchored to.
    pub channel_id: String,
    /// Role names the member holds.
    pub roles: Vec<String>,
    /// Continue this conversation, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<Uuid>,
    /// The question.
    pub message: String,
    /// Who may read the turn.
    pub visibility: Visibility,
    /// Continue the latest open conversation of this user in channel_id with this visibility.
    pub continue_channel: bool,
    /// The bot message this one replies to, when it replies to one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
    /// Images attached to the message.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Attachment>,
    /// Other files attached to the message, opened in the engine sandbox.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileAttachment>,
}

/// An action the engine is holding until the caller who asked answers it.
#[derive(Debug, Clone, Deserialize)]
pub struct Confirmation {
    /// Single-use token the buttons echo back to identify the held action.
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
    /// Sources the answer rests on, best first.
    #[serde(default)]
    pub citations: Vec<Citation>,
    /// Set when the engine stopped to ask.
    #[serde(default)]
    pub confirmation: Option<Confirmation>,
    /// How the run ended.
    pub status: String,
    /// Remembered things the answer drew on, one line each.
    #[serde(default)]
    pub memories: Vec<String>,
}

/// One source under an answer. Mirrors engine::core::types::knowledge::evidence::Citation.
#[derive(Debug, Clone, Deserialize)]
pub struct Citation {
    /// Human-readable source name.
    pub title: String,
    /// Canonical page URL, when the source has one.
    #[serde(default)]
    pub url: Option<String>,
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
    /// Who may read the turn the action belongs to.
    pub visibility: Visibility,
}
