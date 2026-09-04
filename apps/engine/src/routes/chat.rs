//! `POST /chat`: one user message in, one answer with citations out.

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::agent::harness::agent::Agent;
use crate::core::traits::conversation::ConversationStore;
use crate::core::types::agent::AgentError;
use crate::core::types::chat::{ChatRequest, ChatResponse, ErrorBody};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;

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

fn error(status: StatusCode, ctx: &RequestContext, text: &str) -> Response {
    (
        status,
        Json(ErrorBody {
            request_id: ctx.request_id,
            error: text.into(),
        }),
    )
        .into_response()
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
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            &ctx,
            "conversation store unavailable",
        );
    }
    match state.agent.run(&ctx, &req.message).await {
        Ok(answer) => Json(ChatResponse {
            request_id: ctx.request_id,
            conversation_id: ctx.conversation_id,
            citations: answer.evidence.iter().map(Evidence::citation).collect(),
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
            error(StatusCode::BAD_GATEWAY, &ctx, "the model is unavailable")
        }
        Err(AgentError::Store(e)) => {
            tracing::error!(error = %e, request_id = %ctx.request_id, "store failed");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                &ctx,
                "a store is unavailable",
            )
        }
    }
}
