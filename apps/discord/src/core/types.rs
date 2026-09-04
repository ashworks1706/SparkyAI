//! Every struct in the bot: the `/chat` wire mirror, the engine client, and the event handler.
//! Behaviour lives in `engine_client` and `bot`.

use std::collections::HashMap;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serenity::all::{GuildId, UserId};
use tokio::sync::Mutex;
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

/// A pending confirmation the engine wants the user to approve.
#[derive(Debug, Clone, Deserialize)]
pub struct Confirmation {
    /// Single-use token.
    pub token: Uuid,
    /// Tool that would run.
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

/// HTTP client bound to one engine.
#[derive(Debug, Clone)]
pub struct EngineClient {
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) token: SecretString,
}

/// Per-process bot state and the serenity event handler.
pub struct Handler {
    pub(crate) engine: EngineClient,
    pub(crate) guild_id: GuildId,
    /// Conversation each user is continuing. Lost on restart; `/reset` clears it.
    pub(crate) conversations: Mutex<HashMap<UserId, Uuid>>,
}
