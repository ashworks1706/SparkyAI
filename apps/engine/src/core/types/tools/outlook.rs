//! Outlook read models and the errors a Microsoft Graph call returns.

use serde::{Deserialize, Serialize};

/// One event on the caller's Outlook calendar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlookEvent {
    /// Event subject.
    pub subject: String,
    /// When it starts, as Graph returned it.
    pub start: Option<String>,
    /// Where it is, if given.
    pub location: Option<String>,
    /// Link to the event.
    pub url: Option<String>,
}

/// One message in the caller's Outlook mailbox.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlookMessage {
    /// Message subject.
    pub subject: String,
    /// Who it is from, if given.
    pub from: Option<String>,
    /// When it arrived, as Graph returned it.
    pub received: Option<String>,
    /// The opening of the body.
    pub preview: Option<String>,
    /// Link to the message.
    pub url: Option<String>,
}

/// A Graph request that could not be answered.
#[derive(Debug, thiserror::Error)]
pub enum OutlookError {
    /// Microsoft Graph could not be reached.
    #[error("graph is unreachable: {0}")]
    Unreachable(String),
    /// Graph answered with an error status.
    #[error("graph returned {0}")]
    Refused(u16),
    /// Graph answered with something other than the expected shape.
    #[error("graph returned an unexpected response: {0}")]
    Malformed(String),
}
