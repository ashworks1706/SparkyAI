//! Typed HTTP client for the engine's `/chat`. Service token auth.

use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};

use crate::core::types::{ChatRequest, ChatResponse, EngineError};

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
