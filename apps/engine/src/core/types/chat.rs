//! `ChatRequest`, `ChatResponse`, `ErrorBody` — the `/chat` wire contract with clients.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::types::policy::ConfirmationRequest;
use crate::core::types::tool::ToolRun;
use crate::core::types::trace::RunStatus;

/// Request body.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Caller id as the edge knows it.
    pub user_id: String,
    /// Tenant scope; defaults to the configured guild.
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// Channel the message came from.
    #[serde(default = "default_channel")]
    pub channel_id: String,
    /// Roles asserted by the edge.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Continue this conversation; omit to start one.
    #[serde(default)]
    pub conversation_id: Option<Uuid>,
    /// The message.
    pub message: String,
}

fn default_channel() -> String {
    "http".into()
}

/// Response body.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    /// Trace id.
    pub request_id: Uuid,
    /// Conversation to continue with.
    pub conversation_id: Uuid,
    /// The answer.
    pub text: String,
    /// Citation lines, best first.
    pub citations: Vec<String>,
    /// Set when the agent stopped to ask.
    pub confirmation: Option<ConfirmationRequest>,
    /// How the run ended.
    pub status: RunStatus,
    /// Model calls made.
    pub steps: u32,
    /// Tools that ran, in order.
    pub tools: Vec<ToolRun>,
    /// Total tokens.
    pub tokens: u32,
    /// Estimated cost in USD.
    pub cost_usd: f64,
}

/// Error body.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    /// Trace id.
    pub request_id: Uuid,
    /// What went wrong.
    pub error: String,
    /// HTTP status this would have carried. Present on the stream, where the frame is the only
    /// place a client can read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
}
