//! POST /profile/forget removes what the graph holds about one user. POST /profile/list shows it.

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use secrecy::SecretString;

use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::profile::{
    ForgetRequest, ForgetResponse, ListRequest, ListResponse, ListedNode, ListedRelation,
    ProfileError,
};
use crate::routes::chat::{authorized, too_many};
use crate::routes::rate_limit::RateLimiter;

/// What the profile routes need.
#[derive(Clone)]
pub struct ProfileState {
    /// The graph, when profile recording is on.
    pub graph: Option<Arc<dyn ProfileGraph>>,
    /// Nodes and relations a list returns, each.
    pub list_limit: usize,
    /// Wall-clock budget for one call.
    pub request_budget: Duration,
    /// Per-user request limit, shared with the chat routes.
    pub rate_limit: RateLimiter,
    /// Tenant used when the client sends none.
    pub default_tenant: String,
    /// Bearer token every caller must present.
    pub service_token: SecretString,
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
    if !state.rate_limit.allow(&req.user_id) {
        return too_many(&req.user_id);
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
    // The request context scopes the delete to the caller.
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let ctx = RequestContext::new(tenant, req.user_id, state.request_budget);
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

/// Shows the nodes and relations of the caller's graph.
pub async fn list(
    State(state): State<ProfileState>,
    headers: HeaderMap,
    Json(req): Json<ListRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    if !state.rate_limit.allow(&req.user_id) {
        return too_many(&req.user_id);
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
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let ctx = RequestContext::new(tenant, req.user_id, state.request_budget);
    let listed = async {
        let nodes = graph.recall(&ctx, state.list_limit).await?;
        let relations = graph.relations(&ctx, state.list_limit).await?;
        Ok::<_, ProfileError>((nodes, relations))
    }
    .await;
    match listed {
        Ok((nodes, relations)) => Json(ListResponse {
            nodes: nodes.into_iter().map(ListedNode::from).collect(),
            relations: relations.into_iter().map(ListedRelation::from).collect(),
        })
        .into_response(),
        Err(error) => {
            tracing::error!(error = %error, "profile list failed");
            (StatusCode::SERVICE_UNAVAILABLE, "could not reach the graph").into_response()
        }
    }
}
