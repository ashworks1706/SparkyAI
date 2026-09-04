//! HTTP client for the engine's `/chat`.

use std::time::{Duration, Instant};

use secrecy::{ExposeSecret, SecretString};
use tokio::sync::mpsc::UnboundedSender;

use crate::core::types::{ChatRequest, ChatResponse, Event};

/// Client bound to one engine.
#[derive(Clone)]
pub struct EngineClient {
    http: reqwest::Client,
    url: String,
    token: Option<SecretString>,
}

impl EngineClient {
    /// A client for the engine at `base_url`.
    pub fn new(base_url: &str, token: Option<SecretString>) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_mins(3))
            .build()?;
        Ok(Self {
            http,
            url: format!("{}/chat", base_url.trim_end_matches('/')),
            token,
        })
    }

    /// Sends one turn in the background; the reply arrives as an [`Event::ChatReply`].
    pub fn send(&self, req: ChatRequest, tx: UnboundedSender<Event>) {
        let this = self.clone();
        tokio::spawn(async move {
            let started = Instant::now();
            let result = this.chat(&req).await;
            let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            let _ = tx.send(Event::ChatReply { latency_ms, result });
        });
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, String> {
        let mut request = self.http.post(&self.url).json(req);
        if let Some(t) = &self.token {
            request = request.bearer_auth(t.expose_secret());
        }
        let response = request
            .send()
            .await
            .map_err(|e| format!("engine unreachable at {}: {e}", self.url))?;
        let status = response.status();
        let body = response.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!(
                "engine returned {status}: {}",
                body.chars().take(300).collect::<String>()
            ));
        }
        serde_json::from_str(&body).map_err(|e| format!("bad reply body: {e}"))
    }
}
