//! Typed HTTP client for the engine's `/chat`. Service token auth.

use std::time::Duration;

use opentelemetry::trace::TraceContextExt;
use secrecy::{ExposeSecret, SecretString};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::core::types::{ChatRequest, ChatResponse, EngineError, ErrorFrame, Progress, Update};
use crate::sse::drain_frames;

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

    /// Runs one chat turn, reporting progress on `tx` until the answer or a failure arrives.
    /// Exactly one `Answer` or `Failed` is sent last.
    pub async fn chat_stream(&self, req: &ChatRequest, tx: UnboundedSender<Update>) {
        let mut request = self
            .http
            .post(format!("{}/chat/stream", self.base_url))
            .bearer_auth(self.token.expose_secret())
            .json(req);
        if let Some(traceparent) = current_traceparent() {
            request = request.header("traceparent", traceparent);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => {
                let _ = tx.send(Update::Failed(EngineError::Transport(e.to_string())));
                return;
            }
        };
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let _ = tx.send(Update::Failed(EngineError::Status {
                status: status.as_u16(),
                body: body.chars().take(300).collect(),
            }));
            return;
        }

        let mut buf = String::new();
        let mut body = response.bytes_stream();
        let mut answered = false;
        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(bytes) => bytes,
                Err(e) => {
                    if !answered {
                        let _ = tx.send(Update::Failed(EngineError::Transport(e.to_string())));
                    }
                    return;
                }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            for (name, data) in drain_frames(&mut buf) {
                match name.as_str() {
                    "progress" => match serde_json::from_str::<Progress>(&data) {
                        Ok(p) => {
                            let _ = tx.send(Update::Progress(p.text));
                        }
                        Err(e) => tracing::debug!(error = %e, "unreadable progress frame"),
                    },
                    "answer" => match serde_json::from_str::<ChatResponse>(&data) {
                        Ok(answer) => {
                            answered = true;
                            let _ = tx.send(Update::Answer(Box::new(answer)));
                        }
                        Err(e) => {
                            answered = true;
                            let _ = tx.send(Update::Failed(EngineError::Transport(format!(
                                "bad body: {e}"
                            ))));
                        }
                    },
                    "error" => {
                        answered = true;
                        // The frame carries the status the JSON route would have used; without
                        // it a capacity 503 would be indistinguishable from an outage.
                        let frame = serde_json::from_str::<ErrorFrame>(&data);
                        let (status, body) = match &frame {
                            Ok(f) => (f.status.unwrap_or(502), f.error.clone()),
                            Err(_) => (502, data.chars().take(300).collect()),
                        };
                        let _ = tx.send(Update::Failed(EngineError::Status { status, body }));
                    }
                    _ => {}
                }
            }
        }
        if !answered {
            let _ = tx.send(Update::Failed(EngineError::Transport(
                "the engine closed the stream without answering".into(),
            )));
        }
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
