//! Per-user OAuth login: POST /oauth/{provider}/authorize mints a consent URL for the bot, and
//! GET /oauth/{provider}/callback receives the code, exchanges it, and stores the grant.

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
use crate::runtime::tools::oauth::CanvasOAuthClient;

/// The only provider a login serves today.
pub const CANVAS: &str = "canvas";

/// What the OAuth routes read.
#[derive(Clone)]
pub struct OAuthState {
    /// Where grants and pending logins are held.
    pub store: Arc<dyn OAuthStore>,
    /// The Canvas client, when canvas login is enabled.
    pub canvas: Option<Arc<CanvasOAuthClient>>,
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
    if provider != CANVAS {
        return (StatusCode::NOT_FOUND, "unknown provider").into_response();
    }
    let Some(canvas) = &state.canvas else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "canvas login is not enabled",
        )
            .into_response();
    };
    let token = Uuid::new_v4().to_string();
    if let Err(error) = state
        .store
        .begin_consent(&token, USER_SCOPE, &req.user, CANVAS, state.state_ttl)
        .await
    {
        tracing::error!(%error, "could not start a canvas login");
        return (StatusCode::INTERNAL_SERVER_ERROR, "could not start login").into_response();
    }
    Json(AuthorizeResponse {
        url: canvas.authorize_url(&token),
    })
    .into_response()
}

/// Receives the provider redirect, exchanges the code, and stores the grant. No service token.
pub async fn callback(
    State(state): State<OAuthState>,
    Path(provider): Path<String>,
    Query(query): Query<CallbackQuery>,
) -> Response {
    if provider != CANVAS {
        return page("Unknown provider.");
    }
    let Some(canvas) = &state.canvas else {
        return page("Canvas login is not enabled.");
    };
    if let Some(error) = query.error.as_deref() {
        tracing::info!(%error, "canvas login refused at the provider");
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
            tracing::error!(%error, "could not read a canvas login state");
            return page("Something went wrong finishing the login. Run /login again.");
        }
    };
    if consent.provider != CANVAS {
        return page("This login link was for another provider.");
    }
    let tokens = match canvas.exchange(&code).await {
        Ok(tokens) => tokens,
        Err(error) => {
            tracing::warn!(%error, "canvas code exchange failed");
            return page("Canvas would not complete the login. Run /login again.");
        }
    };
    if let Err(error) = state
        .store
        .save_grant(&consent.tenant_id, &consent.user_id, CANVAS, &tokens)
        .await
    {
        tracing::error!(%error, "could not save a canvas grant");
        return page("Could not save your connection. Run /login again.");
    }
    page("You are connected. Return to Discord and ask me about your Canvas.")
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
    if provider != CANVAS {
        return (StatusCode::NOT_FOUND, "unknown provider").into_response();
    }
    match state
        .store
        .delete_grant(USER_SCOPE, &req.user, CANVAS)
        .await
    {
        Ok(removed) => Json(DisconnectResponse { removed }).into_response(),
        Err(error) => {
            tracing::error!(%error, "could not delete a canvas grant");
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
