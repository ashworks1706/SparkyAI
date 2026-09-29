//! Outlook (Microsoft Graph) read API.

use async_trait::async_trait;
use secrecy::SecretString;

use crate::core::types::tools::outlook::{OutlookError, OutlookEvent, OutlookMessage};

/// Read-only access to a caller's Outlook calendar and mail on Microsoft Graph.
#[async_trait]
pub trait Outlook: Send + Sync {
    /// The caller's upcoming calendar events, soonest first.
    async fn calendar(&self, token: &SecretString) -> Result<Vec<OutlookEvent>, OutlookError>;

    /// The caller's most recent mail, newest first.
    async fn mail(&self, token: &SecretString) -> Result<Vec<OutlookMessage>, OutlookError>;
}
