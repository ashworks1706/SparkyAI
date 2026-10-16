//! Per-user account links: authorize hands out the store's login link; disconnect removes a grant.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use secrecy::SecretString;

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::http::oauth::{AuthorizeRequest, AuthorizeResponse, DisconnectResponse};
use crate::core::types::tools::oauth::USER_SCOPE;
use crate::routes::auth::authorized;

/// What the OAuth routes read.
#[derive(Clone)]
pub struct OAuthState {
    /// Where grants are held and logins start.
    pub store: Arc<dyn OAuthStore>,
    /// Bearer token both routes require.
    pub service_token: SecretString,
}

/// Hands out the login link for a provider. Requires the service token.
pub async fn authorize(
    State(state): State<OAuthState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(req): Json<AuthorizeRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    match state.store.login_link(&req.user, &provider).await {
        Ok(Some(url)) => Json(AuthorizeResponse { url }).into_response(),
        Ok(None) => not_enabled(),
        Err(error) => {
            tracing::error!(%error, %provider, "could not start a login");
            (StatusCode::INTERNAL_SERVER_ERROR, "could not start login").into_response()
        }
    }
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

/// The answer when no login is offered for a provider.
fn not_enabled() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        "login is not enabled for this provider",
    )
        .into_response()
}
