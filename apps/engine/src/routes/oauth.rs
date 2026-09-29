//! Per-user OAuth login: authorize mints a consent URL and callback exchanges the code.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Json, response};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::tools::oauth::USER_SCOPE;
use crate::routes::chat::authorized;
use crate::runtime::tools::oauth::WebOAuthClient;

/// What the OAuth routes read.
#[derive(Clone)]
pub struct OAuthState {
    /// Where grants and pending logins are held.
    pub store: Arc<dyn OAuthStore>,
    /// The web OAuth client per enabled provider, by provider key.
    pub providers: HashMap<String, Arc<WebOAuthClient>>,
    /// Bearer token the authorize route requires.
    pub service_token: SecretString,
    /// How long a pending login stays valid.
    pub state_ttl: Duration,
}

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

/// Mints a consent URL for a caller. Requires the service token.
pub async fn authorize(
    State(state): State<OAuthState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(req): Json<AuthorizeRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let Some(client) = state.providers.get(&provider) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "login is not enabled for this provider",
        )
            .into_response();
    };
    let token = Uuid::new_v4().to_string();
    if let Err(error) = state
        .store
        .begin_consent(&token, USER_SCOPE, &req.user, &provider, state.state_ttl)
        .await
    {
        tracing::error!(%error, %provider, "could not start a login");
        return (StatusCode::INTERNAL_SERVER_ERROR, "could not start login").into_response();
    }
    Json(AuthorizeResponse {
        url: client.authorize_url(&token),
    })
    .into_response()
}

/// Receives the provider redirect, exchanges the code, and stores the grant. No service token.
pub async fn callback(
    State(state): State<OAuthState>,
    Path(provider): Path<String>,
    Query(query): Query<CallbackQuery>,
) -> Response {
    let Some(client) = state.providers.get(&provider) else {
        return page("Login is not enabled for this provider.");
    };
    if let Some(error) = query.error.as_deref() {
        tracing::info!(%error, %provider, "login refused at the provider");
        return page("The login was cancelled or refused. You can close this tab.");
    }
    let (Some(code), Some(login)) = (query.code, query.state) else {
        return page("The login link was incomplete. Run /login again.");
    };
    let consent = match state.store.take_consent(&login).await {
        Ok(Some(consent)) => consent,
        Ok(None) => {
            return page("This login link has expired or was already used. Run /login again.");
        }
        Err(error) => {
            tracing::error!(%error, "could not read a login state");
            return page("Something went wrong finishing the login. Run /login again.");
        }
    };
    if consent.provider != provider {
        return page("This login link was for another provider.");
    }
    let tokens = match client.exchange(&code).await {
        Ok(tokens) => tokens,
        Err(error) => {
            tracing::warn!(%error, %provider, "code exchange failed");
            return page("The provider would not complete the login. Run /login again.");
        }
    };
    if let Err(error) = state
        .store
        .save_grant(&consent.tenant_id, &consent.user_id, &provider, &tokens)
        .await
    {
        tracing::error!(%error, "could not save a grant");
        return page("Could not save your connection. Run /login again.");
    }
    page("You are connected. Return to Discord and ask me in a direct message.")
}

/// Removes a caller's grant. Requires the service token.
pub async fn disconnect(
    State(state): State<OAuthState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(req): Json<AuthorizeRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    if !state.providers.contains_key(&provider) {
        return (StatusCode::NOT_FOUND, "unknown provider").into_response();
    }
    match state
        .store
        .delete_grant(USER_SCOPE, &req.user, &provider)
        .await
    {
        Ok(removed) => Json(DisconnectResponse { removed }).into_response(),
        Err(error) => {
            tracing::error!(%error, %provider, "could not delete a grant");
            (StatusCode::INTERNAL_SERVER_ERROR, "could not disconnect").into_response()
        }
    }
}

/// A minimal HTML page carrying one line to the user's browser.
fn page(message: &str) -> Response {
    let body = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>SparkyAI</title></head>\
         <body style=\"font-family: system-ui, sans-serif; max-width: 32rem; margin: 4rem auto; \
         padding: 0 1rem; line-height: 1.5\"><h1>SparkyAI</h1><p>{message}</p></body></html>"
    );
    (StatusCode::OK, response::Html(body)).into_response()
}
