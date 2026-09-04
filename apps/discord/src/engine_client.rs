//! Typed HTTP client for the engine's `/chat`. Service token auth.

use std::time::Duration;

use opentelemetry::trace::TraceContextExt;
use secrecy::{ExposeSecret, SecretString};
use tracing_opentelemetry::OpenTelemetrySpanExt;

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

    /// Runs one chat turn. Carries the current span's trace context so the engine's spans
    /// land in the same Phoenix trace as the interaction.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, EngineError> {
        let mut request = self
            .http
            .post(format!("{}/chat", self.base_url))
            .bearer_auth(self.token.expose_secret())
            .json(req);
        if let Some(traceparent) = current_traceparent() {
            request = request.header("traceparent", traceparent);
        }
        let response = request
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

/// W3C `traceparent` for the current span, if tracing is exporting.
pub fn current_traceparent() -> Option<String> {
    let cx = tracing::Span::current().context();
    let sc = cx.span().span_context().clone();
    sc.is_valid().then(|| {
        format!(
            "00-{}-{}-{:02x}",
            sc.trace_id(),
            sc.span_id(),
            sc.trace_flags().to_u8()
        )
    })
}
