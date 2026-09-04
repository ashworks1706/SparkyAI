//! `UserEvent` — the normalized inbound message every adapter produces.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Which edge produced the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventSource {
    /// A Discord slash command or message.
    Discord,
    /// The HTTP chat API (web, admin, tests).
    Http,
}

/// One inbound user message, before it becomes a `RequestContext` and a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEvent {
    /// Correlates with `RequestContext::request_id`.
    pub request_id: Uuid,
    /// Where it came from.
    pub source: EventSource,
    /// Scope the data belongs to (the Discord guild).
    pub tenant_id: String,
    /// The caller.
    pub user_id: String,
    /// Channel or thread it arrived in.
    pub channel_id: String,
    /// Roles asserted by the edge adapter.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Conversation to continue; `None` starts a new one.
    #[serde(default)]
    pub conversation_id: Option<Uuid>,
    /// The message text.
    pub content: String,
    /// When the edge received it.
    pub received_at: DateTime<Utc>,
}
