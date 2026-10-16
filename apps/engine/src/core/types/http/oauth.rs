//! Wire shapes of the per-user account link routes.

use serde::{Deserialize, Serialize};

/// The caller a login belongs to, sent by the bot.
#[derive(Deserialize)]
pub struct AuthorizeRequest {
    /// The caller who started the login.
    pub user: String,
}

/// The login link the caller opens.
#[derive(Serialize)]
pub struct AuthorizeResponse {
    /// The link that starts the provider login.
    pub url: String,
}

/// Whether a disconnect removed a stored grant.
#[derive(Serialize)]
pub struct DisconnectResponse {
    /// True when a grant was there and is now gone.
    pub removed: bool,
}
