//! Google Calendar read API.

use async_trait::async_trait;
use secrecy::SecretString;

use crate::core::types::tools::gcal::{GCalError, GCalEvent};

/// Read-only access to a caller's Google calendar.
#[async_trait]
pub trait GoogleCalendar: Send + Sync {
    /// The caller's upcoming events on their primary calendar, soonest first.
    async fn events(&self, token: &SecretString) -> Result<Vec<GCalEvent>, GCalError>;
}
