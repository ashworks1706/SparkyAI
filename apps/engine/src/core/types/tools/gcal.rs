//! Google Calendar read models and the errors a Calendar API call returns.

use serde::{Deserialize, Serialize};

/// One event on the caller's Google calendar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GCalEvent {
    /// Event summary.
    pub summary: String,
    /// When it starts, as the API returned it (a date-time or an all-day date).
    pub start: Option<String>,
    /// Where it is, if given.
    pub location: Option<String>,
    /// Link to the event.
    pub url: Option<String>,
}

/// A Google Calendar request that could not be answered.
#[derive(Debug, thiserror::Error)]
pub enum GCalError {
    /// The Calendar API could not be reached.
    #[error("google calendar is unreachable: {0}")]
    Unreachable(String),
    /// The API answered with an error status.
    #[error("google calendar returned {0}")]
    Refused(u16),
    /// The API answered with something other than the expected shape.
    #[error("google calendar returned an unexpected response: {0}")]
    Malformed(String),
}
