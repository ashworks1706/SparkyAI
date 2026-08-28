//! `RequestContext` — per-request state threaded through every call.

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Per-request state. Created at the edge (Discord, HTTP) and threaded
/// through every model call, tool call, and trace event. Never global.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique id for this request; the trace id.
    pub request_id: Uuid,
    /// Caller identity as known to the edge adapter.
    pub user_id: String,
    /// When the request entered the system.
    pub started_at: DateTime<Utc>,
}

impl RequestContext {
    /// Creates a context with a fresh `request_id`.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            user_id: user_id.into(),
            started_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_are_distinct_per_request() {
        let a = RequestContext::new("u1");
        let b = RequestContext::new("u1");
        assert_ne!(a.request_id, b.request_id);
    }
}
