//! Wire shapes for the `OpenAI`-compatible surface, so off-the-shelf chat clients can drive the
//! agent. Only the fields the engine honours are modelled; the rest are ignored on the way in.

use serde::{Deserialize, Serialize};

/// One turn as an `OpenAI` client sends or receives it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// Message text. Clients may send null for tool-only turns.
    #[serde(default, deserialize_with = "null_as_empty")]
    pub content: String,
}

fn null_as_empty<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

/// `POST /v1/chat/completions` body.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// Full history as the client knows it; the engine reads the newest user turn.
    pub messages: Vec<ChatMessage>,
    /// Clients that want server-sent events. The engine sends the answer as one event.
    #[serde(default)]
    pub stream: bool,
    /// Caller identity, used to keep one client's chats apart.
    #[serde(default)]
    pub user: Option<String>,
}

/// `POST /v1/chat/completions` response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    /// Completion id; carries the engine's request id so it can be found in a trace.
    pub id: String,
    /// Always `chat.completion`.
    pub object: &'static str,
    /// Unix seconds.
    pub created: i64,
    /// Model name the engine answered as.
    pub model: String,
    /// One choice; the engine does not sample alternatives.
    pub choices: Vec<Choice>,
    /// Token totals across every model call in the run.
    pub usage: CompletionUsage,
}

/// One completion choice.
#[derive(Debug, Serialize)]
pub struct Choice {
    /// Always 0.
    pub index: u32,
    /// The assistant turn.
    pub message: ChatMessage,
    /// `stop`, or the run status when it ended some other way.
    pub finish_reason: String,
}

/// Token totals in `OpenAI`'s shape.
#[derive(Debug, Serialize)]
#[allow(
    clippy::struct_field_names,
    reason = "the field names are OpenAI's wire contract"
)]
pub struct CompletionUsage {
    /// Prompt tokens across the run.
    pub prompt_tokens: u32,
    /// Generated tokens across the run.
    pub completion_tokens: u32,
    /// Sum of both.
    pub total_tokens: u32,
}

/// `GET /v1/models` response.
#[derive(Debug, Serialize)]
pub struct ModelList {
    /// Always `list`.
    pub object: &'static str,
    /// The single model the engine answers as.
    pub data: Vec<ModelCard>,
}

/// One entry in `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelCard {
    /// Model name.
    pub id: String,
    /// Always `model`.
    pub object: &'static str,
    /// Unix seconds.
    pub created: i64,
    /// Always `sparky`.
    pub owned_by: &'static str,
}
