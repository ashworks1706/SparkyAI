//! `POST /chat`: one user message in, one answer with citations out.

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::harness::agent::{Agent, AgentError};
use crate::agent::harness::conversation::ConversationStore;
use crate::agent::harness::policy::ConfirmationRequest;
use crate::agent::harness::trace::RunStatus;
use crate::core::types::context::RequestContext;

/// What the chat route needs.
#[derive(Clone)]
pub struct ChatState {
    /// The agent.
    pub agent: Agent,
    /// To create the conversation row before the run.
    pub conversations: Option<Arc<dyn ConversationStore>>,
    /// Per-request wall-clock budget.
    pub request_budget: Duration,
    /// Tenant used when the client sends none (single-guild deployments).
    pub default_tenant: String,
}

/// Request body.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Caller id as the edge knows it.
    pub user_id: String,
    /// Tenant scope; defaults to the configured guild.
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// Channel the message came from.
    #[serde(default = "default_channel")]
    pub channel_id: String,
    /// Roles asserted by the edge.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Continue this conversation; omit to start one.
    #[serde(default)]
    pub conversation_id: Option<Uuid>,
    /// The message.
    pub message: String,
}

fn default_channel() -> String {
    "http".into()
}

/// Response body.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    /// Trace id.
    pub request_id: Uuid,
    /// Conversation to continue with.
    pub conversation_id: Uuid,
    /// The answer.
    pub text: String,
    /// Citation lines, best first.
    pub citations: Vec<String>,
    /// Set when the agent stopped to ask.
    pub confirmation: Option<ConfirmationRequest>,
    /// How the run ended.
    pub status: RunStatus,
    /// Model calls made.
    pub steps: u32,
    /// Total tokens.
    pub tokens: u32,
    /// Estimated cost in USD.
    pub cost_usd: f64,
}

/// Error body.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    /// Trace id.
    pub request_id: Uuid,
    /// What went wrong.
    pub error: String,
}

/// Handles one chat turn.
pub async fn chat(State(state): State<ChatState>, Json(req): Json<ChatRequest>) -> Response {
    if req.message.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "message is empty").into_response();
    }
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let mut ctx =
        RequestContext::new(tenant, req.user_id, state.request_budget).with_roles(req.roles);
    if let Some(id) = req.conversation_id {
        ctx = ctx.with_conversation(id);
    }
    if let Some(store) = &state.conversations
        && let Err(e) = store.ensure(&ctx, &req.channel_id).await
    {
        tracing::error!(error = %e, "conversation store unavailable");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                request_id: ctx.request_id,
                error: "conversation store unavailable".into(),
            }),
        )
            .into_response();
    }
    match state.agent.run(&ctx, &req.message).await {
        Ok(answer) => Json(ChatResponse {
            request_id: ctx.request_id,
            conversation_id: ctx.conversation_id,
            citations: answer
                .evidence
                .iter()
                .map(crate::core::types::evidence::Evidence::citation)
                .collect(),
            text: answer.text,
            confirmation: answer.confirmation,
            status: answer.status,
            steps: answer.steps,
            tokens: answer.usage.total(),
            cost_usd: answer.cost_usd,
        })
        .into_response(),
        Err(AgentError::Model(e)) => {
            tracing::error!(error = %e, request_id = %ctx.request_id, "model failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorBody {
                    request_id: ctx.request_id,
                    error: "the model is unavailable".into(),
                }),
            )
                .into_response()
        }
        Err(AgentError::Store(e)) => {
            tracing::error!(error = %e, request_id = %ctx.request_id, "store failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody {
                    request_id: ctx.request_id,
                    error: "a store is unavailable".into(),
                }),
            )
                .into_response()
        }
    }
}
