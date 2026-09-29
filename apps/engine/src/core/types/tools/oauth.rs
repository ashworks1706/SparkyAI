//! OAuthTokens and OAuthError: what a per-user grant yields and how obtaining one fails.

use chrono::{DateTime, Utc};
use secrecy::SecretString;

/// Tokens from one grant. Secrets never leave this value except as a bearer header.
#[derive(Debug, Clone)]
pub struct OAuthTokens {
    /// Bearer token for the provider's APIs and MCP servers.
    pub access_token: SecretString,
    /// Token that obtains a new access token. Absent when the grant is not offline.
    pub refresh_token: Option<SecretString>,
    /// Scopes the provider granted.
    pub scopes: Vec<String>,
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

/// The caller and provider a pending consent belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Consent {
    /// The tenant the login was started in.
    pub tenant_id: String,
    /// The caller who started the login.
    pub user_id: String,
    /// The provider being connected.
    pub provider: String,
}

/// Why a grant or a refresh failed. Carries no token and no response body.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OAuthError {
    /// The client is not configured for the request.
    #[error("oauth is not configured: {0}")]
    NotConfigured(String),
    /// The provider could not be reached.
    #[error("the provider could not be reached: {0}")]
    Unreachable(String),
    /// The provider refused the request. Holds the HTTP status and the OAuth error code.
    #[error("the provider refused the token request: HTTP {status} {code}")]
    Refused {
        /// HTTP status of the response.
        status: u16,
        /// The OAuth error code, or empty when the body carried none.
        code: String,
    },
    /// The provider answered with something that is not a token response.
    #[error("the provider sent an unusable token response: {0}")]
    Malformed(String),
    /// The grant cannot be refreshed and the user has to connect again.
    #[error("the grant has no refresh token; connect again")]
    NoRefreshToken,
}
