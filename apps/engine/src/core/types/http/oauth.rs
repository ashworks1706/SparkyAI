//! Wire shapes of the per-user OAuth login routes.

use serde::{Deserialize, Serialize};

/// The caller a login belongs to, sent by the bot.
#[derive(Deserialize)]
pub struct AuthorizeRequest {
    /// The caller who started the login.
    pub user: String,
}

/// The consent URL the caller opens.
#[derive(Serialize)]
pub struct AuthorizeResponse {
    /// The provider consent URL, carrying the login state.
    pub url: String,
}

/// Whether a disconnect removed a stored grant.
#[derive(Serialize)]
pub struct DisconnectResponse {
    /// True when a grant was there and is now gone.
    pub removed: bool,
}

/// The query Canvas returns to the callback.
#[derive(Deserialize)]
pub struct CallbackQuery {
    /// The authorization code, present on success.
    pub code: Option<String>,
    /// The login state minted by authorize.
    pub state: Option<String>,
    /// The error code, present when the user refused or the request was bad.
    pub error: Option<String>,
}
