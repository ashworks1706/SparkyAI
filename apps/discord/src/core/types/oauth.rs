//! Bodies of the per-user OAuth routes.

use serde::{Deserialize, Serialize};

/// Body of POST /oauth/{provider}/authorize and /oauth/{provider}/logout.
#[derive(Debug, Serialize)]
pub struct AuthorizeRequest {
    /// Discord user id.
    pub user: String,
}

/// Reply to POST /oauth/{provider}/authorize.
#[derive(Debug, Deserialize)]
pub struct AuthorizeResponse {
    /// The consent URL the user opens to connect the provider.
    pub url: String,
}

/// Reply to POST /oauth/{provider}/logout.
#[derive(Debug, Deserialize)]
pub struct DisconnectResponse {
    /// Whether a connection was removed.
    pub removed: bool,
}
