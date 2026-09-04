//! Typed HTTP client for the engine's `/chat`. Service token auth.

use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
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
    http: reqwest::Client,
    base_url: String,
    token: SecretString,
}

impl EngineClient {
    /// Builds a client for `base_url`.
    pub fn new(base_url: &str, token: SecretString) -> Result<Self, EngineError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_mins(2))
            .build()
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            token,
        })
    }

    /// Runs one chat turn.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, EngineError> {
        let response = self
            .http
            .post(format!("{}/chat", self.base_url))
            .bearer_auth(self.token.expose_secret())
            .json(req)
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(EngineError::Status {
                status: status.as_u16(),
                body: body.chars().take(300).collect(),
            });
        }
        serde_json::from_str(&body).map_err(|e| EngineError::Transport(format!("bad body: {e}")))
    }
}
