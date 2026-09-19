//! GET /sandbox, DELETE /sandbox/{name}, POST /sandbox/enabled: what the agent is running and
//! the two ways an operator stops it.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::types::tools::sandbox::{SandboxError, SandboxReport};
use crate::routes::chat::authorized;

/// What the sandbox routes need.
#[derive(Clone)]
pub struct SandboxState {
    /// The sandbox, absent when the engine runs without one.
    pub sandbox: Option<Arc<dyn Sandbox>>,
    /// Bearer token every caller must present.
    pub service_token: SecretString,
}

/// What POST /sandbox/enabled takes.
#[derive(Debug, Serialize, Deserialize)]
pub struct Switch {
    /// Whether the agent is offered the tool.
    pub enabled: bool,
}

/// What a kill answers with.
#[derive(Debug, Serialize)]
pub struct Killed {
    /// The container that was removed.
    pub name: String,
}

/// The sessions running now and the commands the agent ran.
pub async fn report(State(state): State<SandboxState>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let Some(sandbox) = &state.sandbox else {
        return Json(SandboxReport {
            enabled: false,
            sessions: Vec::new(),
            commands: Vec::new(),
        })
        .into_response();
    };
    Json(SandboxReport {
        enabled: sandbox.enabled(),
        sessions: sandbox.sessions(),
        commands: sandbox.commands(),
    })
    .into_response()
}

/// Removes one session container. Its workspace goes with it.
pub async fn kill(
    State(state): State<SandboxState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let Some(sandbox) = &state.sandbox else {
        return (StatusCode::NOT_IMPLEMENTED, "this engine runs no sandbox").into_response();
    };
    match sandbox.kill(&name).await {
        Ok(()) => {
            tracing::info!(container = %name, "sandbox session killed by an operator");
            Json(Killed { name }).into_response()
        }
        Err(SandboxError::Refused(reason)) => (StatusCode::NOT_FOUND, reason).into_response(),
        Err(e) => {
            tracing::error!(error = %e, container = %name, "sandbox session was not killed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

/// Offers the tool to the agent, or stops offering it. Containers already up are left alone.
pub async fn switch(
    State(state): State<SandboxState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let req: Switch = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let Some(sandbox) = &state.sandbox else {
        return (StatusCode::NOT_IMPLEMENTED, "this engine runs no sandbox").into_response();
    };
    sandbox.set_enabled(req.enabled);
    tracing::info!(enabled = req.enabled, "sandbox switched by an operator");
    Json(Switch {
        enabled: req.enabled,
    })
    .into_response()
}
