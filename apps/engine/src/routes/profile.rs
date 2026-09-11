//! POST /profile/forget: removes what the graph holds about one user.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::core::traits::profile::ProfileGraph;
use crate::core::types::context::RequestContext;
use crate::routes::chat::authorized;

/// What the profile route needs.
#[derive(Clone)]
pub struct ProfileState {
    /// The graph, when profile recording is on.
    pub graph: Option<Arc<dyn ProfileGraph>>,
    /// Tenant used when the client sends none.
    pub default_tenant: String,
    /// Bearer token every caller must present.
    pub service_token: SecretString,
}

/// Who to forget, and optionally what.
#[derive(Debug, Deserialize)]
pub struct ForgetRequest {
    /// The user whose graph this is.
    pub user_id: String,
    /// Guild the request belongs to.
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// One label to remove. Absent removes everything this user carries.
    #[serde(default)]
    pub label: Option<String>,
}

/// How much went.
#[derive(Debug, Serialize)]
pub struct ForgetResponse {
    /// Nodes removed. Relations through them go with them.
    pub removed: u64,
}

/// Removes one label, or everything, from the caller's graph.
pub async fn forget(
    State(state): State<ProfileState>,
    headers: HeaderMap,
    Json(req): Json<ForgetRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let Some(graph) = &state.graph else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "the profile graph is not enabled",
        )
            .into_response();
    };
    if req.user_id.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "user_id is empty").into_response();
    }
    // The context is what scopes the delete. A caller names only their own user id, and the
    // interface cannot express a delete for anyone else.
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let ctx = RequestContext::new(tenant, req.user_id, std::time::Duration::from_secs(30));
    let removed = match &req.label {
        Some(label) => graph.forget(&ctx, label).await,
        None => graph.forget_all(&ctx).await,
    };
    match removed {
        Ok(removed) => {
            tracing::info!(user = %ctx.user_id, removed, "profile forgotten");
            Json(ForgetResponse { removed }).into_response()
        }
        Err(error) => {
            tracing::error!(error = %error, "profile forget failed");
            (StatusCode::SERVICE_UNAVAILABLE, "could not reach the graph").into_response()
        }
    }
}
