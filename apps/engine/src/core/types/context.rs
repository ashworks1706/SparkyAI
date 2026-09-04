//! `RequestContext` — per-request state threaded through every call.

use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Per-request state. Created at the edge (Discord, HTTP) and threaded
/// through every model call, tool call, and trace event. Never global.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique id for this request; the trace id.
    pub request_id: Uuid,
    /// The Discord guild or other scope the data belongs to.
    pub tenant_id: String,
    /// Caller identity as known to the edge adapter.
    pub user_id: String,
    /// Role names granted to the caller.
    pub roles: Vec<String>,
    /// Conversation this request continues.
    pub conversation_id: Uuid,
    /// Hard stop for the whole request.
    pub deadline: Instant,
    /// Cancelled by the caller or by the deadline.
    pub cancel: CancellationToken,
}

impl RequestContext {
    /// Creates a context with a fresh `request_id` and conversation.
    pub fn new(tenant_id: impl Into<String>, user_id: impl Into<String>, budget: Duration) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            tenant_id: tenant_id.into(),
            user_id: user_id.into(),
            roles: Vec::new(),
            conversation_id: Uuid::new_v4(),
            deadline: Instant::now() + budget,
            cancel: CancellationToken::new(),
        }
    }

    /// Continues an existing conversation.
    pub fn with_conversation(mut self, conversation_id: Uuid) -> Self {
        self.conversation_id = conversation_id;
        self
    }

    /// Grants roles to the caller.
    pub fn with_roles(mut self, roles: Vec<String>) -> Self {
        self.roles = roles;
        self
    }

    /// Whether the caller holds `role`.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Time left before the deadline; zero once passed.
    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}
