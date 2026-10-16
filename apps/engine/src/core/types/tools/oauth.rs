//! OAuthTokens: what a per-user grant yields.

use chrono::{DateTime, Utc};
use secrecy::SecretString;

/// Tokens from one grant. Secrets never leave this value except as a bearer header.
#[derive(Debug, Clone)]
pub struct OAuthTokens {
    /// Bearer token for the provider's APIs and MCP servers.
    pub access_token: SecretString,
    /// When the access token stops working. Absent when the provider does not say.
    pub expires_at: Option<DateTime<Utc>>,
}

impl OAuthTokens {
    /// Whether the access token has passed the moment the provider gave for it.
    #[must_use]
    pub fn expired(&self) -> bool {
        self.expires_at.is_some_and(|at| at <= Utc::now())
    }
}

/// The scope per-user grants are stored under, keyed by the caller not the guild.
pub const USER_SCOPE: &str = "user";
